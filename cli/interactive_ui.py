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
        'create':   'Create a new project',
        'list':     'List all projects',
        'active':   'Set the active project',
        'remove':   'Remove a project',
        'dialogue': 'Start a RAG chat session',
    },
    'corpus': {
        'add':         'Add files to the corpus (--ingest to build index immediately)',
        'update':      'Scan source dirs for new/changed files and re-ingest',
        'remove':      'Remove a document from the corpus',
        'list':        'List corpus documents',
        'restore':     'Restore plain text from text cache or index',
        'provenance':  'Show document provenance and version history',
        'find-source': 'Find source document for a chunk ID',
    },
    'index': {
        'build':  'Build (or rebuild) the search index',
        'status': 'Show search index status',
        'chunks': 'Show indexed chunks for a document',
    },
    'analyze': {
        'search':     'Search corpus (semantic top-k by default; --exact or --threshold)',
        'interpret':  'Synthesise all summaries into a meta-summary',
        'clustering': 'Hierarchical clustering with axial coding',
        'anomaly':    'Detect outlier documents',
        'visualize':  't-SNE scatter plot of embeddings',
        'entity':     'Extract named entities from relevant chunks',
        'categorize': 'Assign chunks to user-defined categories',
        'sentiment':  'Analyse sentiment of relevant chunks',
        'summarize':  'Narrative synthesis of relevant chunks',
        'tools':      'List all available analysis plugins',
    },
    'help': {},
    'quit': {},
}

# Maps short aliases to full command names
ALIASES = {
    'p': 'project',
    'c': 'corpus',
    'i': 'index',
    'v': 'vector',   # deprecated — kept for backwards compat
    'a': 'analyze',
    'q': 'quit',
}

# Per-subcommand flag completions (static, best-effort)
_FLAG_COMPLETIONS = {
    ('analyze', 'search'):     ['--k', '--threshold', '--exact', '--summary'],
    ('analyze', 'interpret'):  ['--k', '--threshold'],
    ('analyze', 'clustering'): ['--k', '--save'],
    ('analyze', 'anomaly'):    ['--k'],
    ('analyze', 'entity'):     ['--k', '--threshold', '--types'],
    ('analyze', 'categorize'): ['--k', '--threshold', '--categories'],
    ('analyze', 'sentiment'):  ['--k', '--threshold'],
    ('analyze', 'summarize'):  ['--k', '--threshold'],
    ('corpus',  'add'):        ['--type', '--ingest'],
    ('corpus',  'update'):     ['--yes', '-y', '--type'],
    ('index',   'build'):      ['--force'],
    ('index',   'chunks'):     ['--meta', '--pretty', '--summary'],
    ('project', 'create'):     [],
    ('project', 'active'):     [],
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
