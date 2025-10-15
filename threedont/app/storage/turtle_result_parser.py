import io
import re

from SPARQLWrapper import SPARQLWrapper, QueryResult

__all__ = ['SPARQLWrapperWithTurtle', 'QueryResultWithTurtle']

VARIABLES_REGEX = re.compile(r"res:binding\s*\[\s*res:variable\s*\"([a-zA-Z0-9_]+)\"\s*;\s*res:value\s*(\S+)\s*\]")
PREFIXES_REGEX = re.compile(r"^@prefix\s+([a-zA-Z0-9_]+):\s*<([^>]+)>\s*\.\s*")


class QueryResultWithTurtle(QueryResult):
    def substitute_prefix(self, var):
        if not ':' in var:
            return var

        if var.startswith("<") and var.endswith(">"):
            return var

        prefix, suffix = var.split(":")
        try:
            return '<' + self._prefixes[prefix] + suffix + '>'
        except KeyError:
            # it's not possible to distinguish between iri with prefix and simple string with a colon
            return var

    def normalize_number(self, value):
        if value.endswith('^^xsd:decimal'):
            return value.split('"')[1]
        return value

    def _convertN3(self):
        encoding = self.response.info().get_content_charset('utf-8')
        self._prefixes = {}
        out = {}

        for line in io.TextIOWrapper(self.response, encoding=encoding):
            prefix = PREFIXES_REGEX.search(line)
            if prefix:
                prefix, uri = prefix.groups()
                self._prefixes[prefix] = uri
                continue

            match = VARIABLES_REGEX.search(line)
            if match:
                var, value = match.groups()
                var = self.substitute_prefix(var)
                value = self.substitute_prefix(value)
                value = self.normalize_number(value)
                if var not in out:
                    out[var] = []
                out[var].append(value)

        return out


class SPARQLWrapperWithTurtle(SPARQLWrapper):
    def query(self):
        return QueryResultWithTurtle(self._query())
