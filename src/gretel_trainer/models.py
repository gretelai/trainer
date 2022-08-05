class _BaseConfig:
    """This class should not be used directly, model-specific configs should be
    derived from this class
    """

    _max_header_clusters: int
    """This should be overridden on concrete classes"""

    # Default values for all configs, may be overridden if needed
    _max_rows: int = 1000000

    # Should be set by concrete constructors
    config_file: str
    max_rows: int
    max_header_clusters: int

    def __init__(
        self,
        config_file: str,
        max_rows: int,
        max_header_clusters: int,
    ):
        self.config_file = config_file
        self.max_rows = max_rows
        self.max_header_clusters = max_header_clusters

        self.validate()

    def validate(self):
        if self.max_rows > self._max_rows:
            raise ValueError("too many rows")

        if self.max_header_clusters > self._max_header_clusters:
            raise ValueError("too many header clusters")


class GretelLSTMConfig(_BaseConfig):

    _max_header_clusters: int = 30

    def __init__(
        self,
        config_file="synthetics/default",
        max_rows=50000,
        max_header_clusters=20,
    ):
        super().__init__(
            config_file=config_file,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
        )


class GretelCTGANConfig(_BaseConfig):

    _max_header_clusters: int = 1000

    def __init__(
        self,
        config_file="synthetics/high-dimensionality",
        max_rows=50000,
        max_header_clusters=500,
    ):
        super().__init__(
            config_file=config_file,
            max_rows=max_rows,
            max_header_clusters=max_header_clusters,
        )