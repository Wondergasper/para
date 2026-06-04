from workers.lang.base_driver import LanguageDriver

_drivers = {}

def register_driver(name: str, driver: LanguageDriver):
    _drivers[name.lower()] = driver

def get_driver(name: str) -> LanguageDriver:
    name_lower = name.lower()
    if name_lower not in _drivers:
        # Lazy imports to avoid circular dependencies during initialization
        if name_lower == "c" or name_lower == "c++":
            from workers.lang.c_driver import CDriver
            register_driver("c", CDriver())
            register_driver("c++", CDriver())
        elif name_lower == "fortran":
            from workers.lang.fortran_driver import FortranDriver
            register_driver("fortran", FortranDriver())
        elif name_lower == "python":
            from workers.lang.python_driver import PythonDriver
            register_driver("python", PythonDriver())
        elif name_lower == "rust":
            from workers.lang.rust_driver import RustDriver
            register_driver("rust", RustDriver())
        else:
            raise ValueError(f"Unsupported language driver: {name}")
    return _drivers[name_lower]
