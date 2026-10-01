def generate_script(context):
    script = []
    script.append({
        "speech": "Let us understand this topic step by step.",
        "draw": "Topic Overview"
    })

    for chunk in context[:3]:
        script.append({
            "speech": chunk[:200],
            "draw": "Key concept"
        })
    return script
