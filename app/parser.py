"""Parser module to parse gear config.json."""

from typing import Tuple
from flywheel_gear_toolkit import GearToolkitContext

def parse_config(
    gear_context: GearToolkitContext,
) -> Tuple[str, bool]:
    """Parses the config info.
    
    Return the requisit inputs and options for the gear.

    Args:
        gear_context: Context.
        
    Returns:
        Tuple of api_key
    """
    api_key = gear_context.get_input("api-key").get("key")
    cde_dict = gear_context.get_input_path("cde-dictionary") 
    
    return api_key, cde_dict


