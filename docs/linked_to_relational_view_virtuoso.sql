CREATE TABLE point (
  s VARCHAR,
  r INTEGER,
  g INTEGER,
  b INTEGER,
  x FLOAT,
  y FLOAT,
  z FLOAT
);

CREATE INDEX idx_point_s ON point (s);

INSERT INTO point (s, r, g, b, x, y, z)
SELECT t."s", t."r", t."g", t."b",t."x", t."y", t."z" FROM (
SPARQL
PREFIX base:<http://www.semanticweb.org/mcodi/ontologies/2024/3/Urban_Ontology#>
PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:<http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?s ?x ?y ?z 
       (COALESCE(?_r, 0) AS ?r)
       (COALESCE(?_g, 0) AS ?g)
       (COALESCE(?_b, 0) AS ?b)
FROM <http://localhost:8890/Nettuno>
WHERE {
    ?s base:X ?x;
        base:Y ?y;
        base:Z ?z.
    OPTIONAL {
        ?s base:R ?_r.
        ?s base:G ?_g.
        ?s base:B ?_b.
    }
}
) AS t;

DB.DBA.RDF_VIEW_CREATE (
    'point_view',                    -- view name
    'point',                         -- SQL table
    '{s}',  -- subject IRI template
    'http://www.semanticweb.org/mcodi/ontologies/2024/3/Urban_Ontology#', -- prefix
    's',                             -- primary key / identifier
    'X', 'x',
    'Y', 'y',
    'Z', 'z',
    'R', 'r',
    'G', 'g',
    'B', 'b'
);

grant select on "DB"."DBA"."point" to SPARQL_SELECT;

SPARQL
PREFIX base: <http://www.semanticweb.org/mcodi/ontologies/2024/3/Urban_Ontology#>

CREATE IRI CLASS base:Subject
    "%U" (in _s VARCHAR NOT NULL) .

;

SPARQL
PREFIX base: <http://www.semanticweb.org/mcodi/ontologies/2024/3/Urban_Ontology#>

ALTER QUAD STORAGE virtrdf:DefaultQuadStorage
FROM "DB"."DBA"."point" AS point_s
{
   CREATE base:qm-point AS GRAPH IRI ("http://localhost:8890/Nettuno") OPTION (exclusive)
   {
      # Subject IRI from column s
      base:Subject(point_s."s")
          base:X point_s."x" ;
          base:Y point_s."y" ;
          base:Z point_s."z" ;
          base:R point_s."r" ;
          base:G point_s."g" ;
          base:B point_s."b" .
   }
}
;
