"""
Tests for utils/device.py — Platform/GPU detection.
"""

import unittest
from unittest.mock import patch, MagicMock
import utils.device as device_module


class TestDetectDevice(unittest.TestCase):
    """Tests for detect_device()."""

    def setUp(self):
        """Reset cached device before each test."""
        self._saved_device = device_module._detected_device
        device_module._detected_device = None

    def tearDown(self):
        device_module._detected_device = self._saved_device

    @patch('utils.device.click')
    def test_config_override_cpu(self, mock_click):
        """Config override 'cpu' should be honoured."""
        result = device_module.detect_device(config_override='cpu')
        self.assertEqual(result, 'cpu')

    @patch('utils.device.click')
    def test_config_override_mps(self, mock_click):
        """Config override 'mps' should be honoured."""
        result = device_module.detect_device(config_override='mps')
        self.assertEqual(result, 'mps')

    @patch('utils.device.click')
    def test_config_override_cuda(self, mock_click):
        """Config override 'cuda' should be honoured."""
        result = device_module.detect_device(config_override='cuda')
        self.assertEqual(result, 'cuda')

    @patch('utils.device.click')
    def test_invalid_override_ignored(self, mock_click):
        """Invalid config override should fall through to auto-detection."""
        with patch.dict('sys.modules', {'torch': MagicMock(
            backends=MagicMock(mps=MagicMock(is_available=MagicMock(return_value=False))),
            cuda=MagicMock(is_available=MagicMock(return_value=False)),
        )}):
            result = device_module.detect_device(config_override='tpu')
            self.assertEqual(result, 'cpu')

    @patch('utils.device.click')
    def test_none_override_auto_detects(self, mock_click):
        """None override should auto-detect."""
        with patch.dict('sys.modules', {'torch': MagicMock(
            backends=MagicMock(mps=MagicMock(is_available=MagicMock(return_value=False))),
            cuda=MagicMock(is_available=MagicMock(return_value=False)),
        )}):
            result = device_module.detect_device(config_override=None)
            self.assertEqual(result, 'cpu')

    @patch('utils.device.click')
    def test_caching(self, mock_click):
        """Second call should return cached value without re-detecting."""
        device_module._detected_device = 'mps'
        result = device_module.detect_device(config_override='cpu')
        # Should return cached 'mps', NOT the override
        self.assertEqual(result, 'mps')

    @patch('utils.device.click')
    def test_auto_detect_mps(self, mock_click):
        """Auto-detection should pick MPS when available."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = device_module.detect_device()
            self.assertEqual(result, 'mps')

    @patch('utils.device.click')
    def test_auto_detect_cuda(self, mock_click):
        """Auto-detection should pick CUDA when MPS unavailable but CUDA is."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = device_module.detect_device()
            self.assertEqual(result, 'cuda')

    @patch('utils.device.click')
    def test_auto_detect_cpu_fallback(self, mock_click):
        """Auto-detection should fall back to CPU when no GPU available."""
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with patch.dict('sys.modules', {'torch': mock_torch}):
            result = device_module.detect_device()
            self.assertEqual(result, 'cpu')


class TestGetDeviceInfo(unittest.TestCase):
    """Tests for get_device_info()."""

    def setUp(self):
        self._saved_device = device_module._detected_device

    def tearDown(self):
        device_module._detected_device = self._saved_device

    def test_returns_dict(self):
        """get_device_info() should return a dictionary."""
        info = device_module.get_device_info()
        self.assertIsInstance(info, dict)

    def test_has_platform_keys(self):
        """Result should contain platform and machine keys."""
        info = device_module.get_device_info()
        self.assertIn('platform', info)
        self.assertIn('machine', info)
        self.assertIn('device', info)

    def test_has_torch_info(self):
        """Result should contain torch version info."""
        info = device_module.get_device_info()
        self.assertIn('torch_version', info)
        self.assertIn('mps_available', info)
        self.assertIn('cuda_available', info)

    def test_device_shows_cached_value(self):
        """device field should reflect cached detection result."""
        device_module._detected_device = 'cuda'
        info = device_module.get_device_info()
        self.assertEqual(info['device'], 'cuda')

    def test_device_not_yet_detected(self):
        """device field should indicate 'not yet detected' before first call."""
        device_module._detected_device = None
        info = device_module.get_device_info()
        self.assertEqual(info['device'], 'not yet detected')


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_valid_devices(self):
        self.assertIn('cpu', device_module.VALID_DEVICES)
        self.assertIn('mps', device_module.VALID_DEVICES)
        self.assertIn('cuda', device_module.VALID_DEVICES)

    def test_device_labels_keys(self):
        for d in device_module.VALID_DEVICES:
            self.assertIn(d, device_module.DEVICE_LABELS)


if __name__ == '__main__':
    unittest.main()
