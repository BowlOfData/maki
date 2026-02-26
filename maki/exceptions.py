"""
Custom exception classes for the Maki framework
"""

class MakiError(Exception):
    """Base exception for all Maki framework errors"""
    pass


class MakiValidationError(MakiError):
    """Exception raised for validation errors"""
    pass


class MakiNetworkError(MakiError):
    """Exception raised for network-related errors"""
    pass


class MakiTimeoutError(MakiNetworkError):
    """Exception raised for timeout errors"""
    pass


class MakiAPIError(MakiError):
    """Exception raised for API response errors"""
    pass