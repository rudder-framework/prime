"""
Configuration schema for classification stability checks.

Add these fields to your typology config.
"""

# Example YAML structure for typology config
STABILITY_CONFIG_SCHEMA = """
typology:
  # Existing settings...
  
  # NEW: Classification stability settings
  stability:
    enabled: true                    # Run stability check
    window_size: 10000               # Rows per window
    tolerance: 0.1                   # Acceptable numeric drift
    flag_on_change: true             # Raise flag if classification differs
    fail_on_change: false            # Hard fail vs soft flag
    
    # Output settings
    output:
      include_in_results: true       # Add stability info to output
      log_differences: true          # Log detailed diff when found
"""

# Python config dataclass
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StabilityConfig:
    """Configuration for classification stability checking."""
    
    enabled: bool = True
    window_size: int = 10000
    tolerance: float = 0.1
    flag_on_change: bool = True
    fail_on_change: bool = False
    include_in_results: bool = True
    log_differences: bool = True
    
    @classmethod
    def from_dict(cls, d: dict) -> "StabilityConfig":
        """Create config from dictionary (e.g., parsed YAML)."""
        stability = d.get("stability", {})
        output = stability.get("output", {})
        
        return cls(
            enabled=stability.get("enabled", True),
            window_size=stability.get("window_size", 10000),
            tolerance=stability.get("tolerance", 0.1),
            flag_on_change=stability.get("flag_on_change", True),
            fail_on_change=stability.get("fail_on_change", False),
            include_in_results=output.get("include_in_results", True),
            log_differences=output.get("log_differences", True),
        )
    
    def to_dict(self) -> dict:
        """Export config to dictionary."""
        return {
            "stability": {
                "enabled": self.enabled,
                "window_size": self.window_size,
                "tolerance": self.tolerance,
                "flag_on_change": self.flag_on_change,
                "fail_on_change": self.fail_on_change,
                "output": {
                    "include_in_results": self.include_in_results,
                    "log_differences": self.log_differences,
                }
            }
        }


@dataclass 
class TypologyConfig:
    """
    Extended typology config with stability settings.
    
    Merge this into your existing typology config structure.
    """
    
    # Existing fields (placeholders - match to your actual config)
    sample_size: int = 10000
    engines: list = field(default_factory=list)
    
    # NEW: Stability checking
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    
    @classmethod
    def from_dict(cls, d: dict) -> "TypologyConfig":
        return cls(
            sample_size=d.get("sample_size", 10000),
            engines=d.get("engines", []),
            stability=StabilityConfig.from_dict(d),
        )
