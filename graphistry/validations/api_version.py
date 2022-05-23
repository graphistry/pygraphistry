import re

class API_Version:

    @staticmethod 
    def validate(version, is_env_var = False):
        entity = "GRAPHISTRY_API_VERSION" if is_env_var else "API"

        if is_env_var:
            # implicitly convert "3" to 3 if version privided by env variable 
            version = int(version) if re.sub(r'\d+', '', version) == '' else version

        # type checking
        if not isinstance(version, int):
            raise TypeError(f"Received {entity} as string value, "f"instead use type: integer")

        # version validation
        if version not in [1,2,3]:
            raise ValueError(f"Recevied invalid API version {value}, Available API versions (1,2,3)")
        
        return True