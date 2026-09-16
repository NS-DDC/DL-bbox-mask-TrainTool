"""Plan a complete model-to-project class map before applying detections."""

from core.project_metadata import validate_class_name


def plan_model_classes(classes: list[dict], names: dict[int, str]):
    if not names or any(type(key) is not int or key < 0 or key >= 65535 for key in names):
        raise ValueError("Model must provide valid class IDs and names.")
    normalized = {key: validate_class_name(value) for key, value in names.items()}
    if len({name.casefold() for name in normalized.values()}) != len(normalized):
        raise ValueError("Model has duplicate class names; an unambiguous mapping is required.")
    planned = [dict(cls) for cls in classes]
    if not planned:
        # Even absent detections and sparse model IDs keep their original slots.
        used_names = {name.casefold() for name in normalized.values()}
        for class_id in range(max(normalized) + 1):
            name = normalized.get(class_id)
            if name is None:
                name = f"class_{class_id}"
                while name.casefold() in used_names:
                    name = "_" + name
            used_names.add(name.casefold())
            planned.append({"name": name, "color": "#00aaff"})
    by_name = {cls["name"].casefold(): index for index, cls in enumerate(planned)}
    mapping = {}
    for model_id, name in sorted(normalized.items()):
        project_id = by_name.get(name.casefold())
        if project_id is None:
            project_id = len(planned)
            planned.append({"name": name, "color": "#00aaff"})
            by_name[name.casefold()] = project_id
        mapping[model_id] = project_id
    return planned, mapping
