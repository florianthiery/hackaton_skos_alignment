
JSKOS Server
Available Endpoints

This service provides a subset of JSKOS API 2.2. All the following endpoints are available on this instance of jskos-server. See documentation on GitHub for details.
General

    GET /status - Return a status object (JSON Schema) 🛈
    GET /checkAuth - Check whether a user is authorized 🛈
    POST/GET /validate - Validate JSKOS data 🛈
    GET /data - Return data for a certain URI or URIs (terminologies, concepts, concordances, mappings, annotations, registries) 🛈

Terminologies (113)

    GET /voc - Return a list of vocabularies (terminologies) 🛈
    GET /voc/top - Return top concepts for a terminology 🛈
    GET /voc/concepts - Return concepts for a terminology 🛈
    GET /voc/suggest - Return terminology suggestions in OpenSearch Suggest Format 🛈
    GET /voc/search - Concept scheme search 🛈

Concepts

    GET /concepts - Return detailed data for concepts or terminologies 🛈
    GET /concepts/narrower - Return narrower concepts for a concept 🛈
    GET /concepts/ancestors - Return ancestor concepts for a concept 🛈
    GET /concepts/suggest - Return concept suggestions 🛈
    GET /concepts/search - Concept search 🛈

Concordances (113)

    GET /concordances - Return a list of concordances for mappings 🛈
    GET /concordances/:_id - Return a specific concordance 🛈
    POST /concordances - Save a concordance in the database 🔒 🛈
    PUT /concordances/:_id - Update a concordance in the database 🔒 🛈
    PATCH /concordances/:_id - Adjust a concordance in the database 🔒 🛈
    DELETE /concordances/:_id - Delete a concordance from the database 🔒 🛈

Mappings (1094072)

    GET /mappings - Return a list of mappings 🛈
    GET /mappings/:_id - Return a specific mapping 🛈
    POST /mappings — Save mappings in the database 🔒 🛈
    PUT /mappings/:_id - Update a mapping in the database 🔒 🛈
    PATCH /mappings/:_id - Adjust a mapping in the database 🔒 🛈
    DELETE /mappings/:_id - Delete a mapping from the database 🔒 🛈
    GET /mappings/infer - Return mappings based on stored mappings and mappings derived by inference 🛈
    POST /mappings/apply - Apply mappings to a set of items 🛈
    GET /mappings/suggest - Suggest notations used in mappings 🛈
    GET /mappings/voc - Return a list of terminologies used in mappings 🛈

Annotations (10845)

    GET /annotations - Return a list of annotations 🛈
    GET /annotations/:_id - Return a specific annotation 🛈
    POST /annotations - Save an annotation in the database 🔒 🛈
    PUT /annotations/:_id - Update an annotation in the database 🔒 🛈
    PATCH /annotations/:_id - Adjust an annotation in the database 🔒 🛈
    DELETE /annotations/:_id - Delete an annotation from the database 🔒 🛈

