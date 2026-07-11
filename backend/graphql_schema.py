"""GraphQL schema for the HPI prediction API (2nd API paradigm, alongside REST).

Reuses the same validate_row / predict_row logic as the REST /api/v1/predict
route, so predictions stay consistent across both paradigms.
"""
from ariadne import QueryType, make_executable_schema, gql

type_defs = gql("""
    type PredictionResult {
        hpi: Float!
    }

    type Query {
        predict(
            year: Int!
            inflation: Float!
            unemployment: Float!
        ): PredictionResult!
    }
""")

query = QueryType()


@query.field("predict")
def resolve_predict(_, info, year, inflation, unemployment):
    # Imported lazily to avoid a circular import with api.py at module load time.
    from backend.api import predict_row
    from backend.preprocessing import validate_row

    row = {"year": year, "inflation": inflation, "unemployment": unemployment}
    validate_row(row)  # raises ValueError -> surfaced as a GraphQL error below
    hpi = predict_row(row)
    return {"hpi": hpi}


schema = make_executable_schema(type_defs, query)
