from fastapi import APIRouter

router = APIRouter()

@router.post("/simplify")
def simplify(query: str):
    return {"message": "Simplify API working", "query": query}
