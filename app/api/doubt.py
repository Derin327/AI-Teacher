from fastapi import APIRouter

router = APIRouter()

@router.post("/doubt")
def doubt(query: str):
    return {"message": "Doubt API working", "query": query}
