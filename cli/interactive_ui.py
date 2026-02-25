"""
prompt_toolkit-powered interactive UI for m3.

Provides autocompletion, persistent history, bottom toolbar, and styled
prompt — all without changing the existing parse/dispatch logic.
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

# ---------------------------------------------------------------------------
# Command tree — used for autocompletion
# ---------------------------------------------------------------------------

COMMAND_TREE = {
    'project': {
        'create':    'Create a new project',
        'list':      'List all projects',
        'active':    'Show or set the active project',
        'remove':    'Remove a project',
        'dialogue':  'Start a RAG dialogue session',
    },
    'corpus': {
        'add':          'Add a document to the corpus',
        'remove':       'Remove a document from the corpus',
        'list':         'List corpus documents',
        'ingest':       'Ingest corpus through the pipeline',
        'summary':      'Show corpus summary',
        'provenance':   'Show document provenance',
        'reconstitute': 'Reconstitute document from chunks',
        'find-source':  'Find source for a passage',
    },
    'vector': {
        'ingest':  'Ingest documents into the vector store',
        'chunks':  'Show vector store chunks',
        'status':  'Show vector store status',
        'query':   'Query the vector store',
        'create':  'Create a new collection',
    },
    'analyze': {
        'topk':   'Top-K semantic search',
        'search': 'Semantic search with threshold',
        'exact':  'Exact keyword search',
        'tools':  'List analysis plugins',
        'run':    'Run an analysis plugin',
    },
    'help': {},
    'quit': {},
}

# Maps short aliases to full command names
ALIASES = {
    'p': 'project',
    'c': 'corpus',
    'v': 'vector',
    'a': 'analyze',
    'q': 'quit',
}

# Per-subcommand flag completions (static, best-effort)
_FLAG_COMPLETIONS = {
    ('analyze', 'topk'):   ['--k', '--save', '--output'],
    ('analyze', 'search'): ['--threshold', '--save', '--output'],
    ('analyze', 'exact'):  ['--save', '--output'],
    ('analyze', 'run'):    ['--plugin', '--output'],
    ('corpus',  'add'):    ['--type'],
    ('corpus',  'ingest'): ['--collection', '--doc-type'],
    ('vector',  'ingest'): ['--collection'],
    ('vector',  'query'):  ['--k', '--collection'],
    ('project', 'create'): [],
    ('project', 'active'): ['--set'],
}


# ---------------------------------------------------------------------------
# Completer
# ---------------------------------------------------------------------------

class M3Completer(Completer):
    """
    Completes m3 slash-commands.

    Trigger character is '/'.  The completer understands three levels:
      Level 1 – top-level group/alias after '/'
      Level 2 – subcommand after '<group> '
      Level 3 – flags after '<group> <subcommand> '
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Only activate when the line starts with '/'
        if not text.startswith('/'):
            return

        # Strip the leading slash
        body = text[1:]
        parts = body.split(' ')

        if len(parts) == 1:
            # Level 1: completing the top-level command/alias
            yield from self._complete_level1(parts[0])

        elif len(parts) == 2:
            # Level 2: completing the subcommand
            group_token = parts[0].lower()
            sub_prefix = parts[1]
            group = ALIASES.get(group_token, group_token)
            yield from self._complete_level2(group, sub_prefix)

        elif len(parts) >= 3:
            # Level 3: completing flags
            group_token = parts[0].lower()
            sub_token = parts[1].lower()
            flag_prefix = parts[-1]
            group = ALIASES.get(group_token, group_token)
            yield from self._complete_level3(group, sub_token, flag_prefix)

    # ------------------------------------------------------------------
    def _complete_level1(self, prefix):
        """Yield top-level command and alias completions."""
        candidates = list(COMMAND_TREE.keys()) + list(ALIASES.keys())
        for name in candidates:
            if name.startswith(prefix):
                # Resolve description: aliases show what they map to
                if name in ALIASES:
                    meta = f'alias for {ALIASES[name]}'
                elif name in ('help', 'quit'):
                    meta = {'help': 'Show help', 'quit': 'Exit m3'}[name]
                else:
                    meta = f'{name} commands'
                yield Completion(
                    name,
                    start_position=-len(prefix),
                    display_meta=meta,
                )

    def _complete_level2(self, group, sub_prefix):
        """Yield subcommand completions for a resolved group name."""
        subcmds = COMMAND_TREE.get(group, {})
        for sub, desc in subcmds.items():
            if sub.startswith(sub_prefix):
                yield Completion(
                    sub,
                    start_position=-len(sub_prefix),
                    display_meta=desc,
                )

    def _complete_level3(self, group, sub, flag_prefix):
        """Yield flag completions for a group+subcommand pair."""
        flags = _FLAG_COMPLETIONS.get((group, sub), [])
        for flag in flags:
            if flag.startswith(flag_prefix):
                yield Completion(
                    flag,
                    start_position=-len(flag_prefix),
                )


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def build_prompt_session(history_path: str) -> PromptSession:
    """
    Build and return a configured PromptSession.

    Parameters
    ----------
    history_path:
        Path to the persistent history file (e.g. ~/.monkey3/history).
    """
    style = Style.from_dict({
        # Prompt tokens (matched against class names in get_styled_prompt)
        'bracket':  '#888888',
        'appname':  '#00aaff bold',
        'colon':    '#888888',
        'project':  '#00ffaa bold',
        'arrow':    '#888888',

        # Completion menu
        'completion-menu.completion':          'bg:#1e1e2e #cdd6f4',
        'completion-menu.completion.current':  'bg:#313244 #89dceb bold',
        'completion-menu.meta.completion':     'bg:#1e1e2e #6c7086',
        'completion-menu.meta.completion.current': 'bg:#313244 #6c7086',

        # Bottom toolbar
        'bottom-toolbar':      'bg:#1e1e2e #6c7086',
        'bottom-toolbar.text': 'bg:#1e1e2e #cdd6f4',
    })

    kb = KeyBindings()

    @kb.add('c-c')
    def _(event):
        raise KeyboardInterrupt

    return PromptSession(
        history=FileHistory(history_path),
        completer=M3Completer(),
        auto_suggest=AutoSuggestFromHistory(),
        style=style,
        key_bindings=kb,
        complete_while_typing=True,
        complete_in_thread=True,
    )


# ---------------------------------------------------------------------------
# Bottom toolbar
# ---------------------------------------------------------------------------

def get_toolbar(session) -> HTML:
    """
    Return the bottom-toolbar content for the PromptSession.

    Parameters
    ----------
    session : M3Session
        The active session object (used to read active_project_name).
    """
    project = session.active_project_name or 'no project active'
    return HTML(
        f'<b>[m3]</b>  project: <b>{project}</b>'
        '   <ansibrightblack>/help · /q to quit · Tab to complete</ansibrightblack>'
    )
