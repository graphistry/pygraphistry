"""Tests for logging behavior across the graphistry package"""
import unittest
import logging
import sys
from io import StringIO


class TestLogging(unittest.TestCase):
    """Test that graphistry's logging doesn't interfere with user logging configurations"""
    
    def test_external_logger_configuration_preserved(self):
        """Regression test for issue #660: graphistry should not modify external loggers"""
        # Set up a logger with custom configuration before importing graphistry
        external_logger = logging.getLogger("user_application")
        handler = logging.StreamHandler()
        original_terminator = handler.terminator
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        external_logger.addHandler(handler)
        
        # Import graphistry (which loads various modules with custom logging)
        import graphistry
        
        # Verify the external logger configuration is unchanged
        self.assertEqual(handler.terminator, original_terminator,
                        "graphistry import modified external handler terminator")
        self.assertEqual(logging.StreamHandler.terminator, '\n',
                        "graphistry import modified global StreamHandler default")
        
        # Verify external logger still works as expected
        capture = StringIO()
        test_handler = logging.StreamHandler(capture)
        test_logger = logging.getLogger("test_logger")
        test_logger.addHandler(test_handler)
        test_logger.setLevel(logging.INFO)
        
        test_logger.info("Line 1")
        test_logger.info("Line 2")
        
        output = capture.getvalue()
        self.assertEqual(output, "Line 1\nLine 2\n",
                        "Logger output should have newlines between messages")


if __name__ == "__main__":
    unittest.main()
