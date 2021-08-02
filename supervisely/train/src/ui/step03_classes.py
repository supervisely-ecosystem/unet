import supervisely_lib as sly


def init(api: sly.Api, data, state, project_id, project_meta: sly.ProjectMeta):
    stats = api.project.get_stats(project_id)
    class_images = {}
    for item in stats["images"]["objectClasses"]:
        class_images[item["objectClass"]["name"]] = item["total"]

    class_objects = {}
    for item in stats["objects"]["items"]:
        class_objects[item["objectClass"]["name"]] = item["total"]

    class_area = {}
    for item in stats["objectsArea"]["items"]:
        class_area[item["objectClass"]["name"]] = round(item["total"], 2)

    # keep only polygon + bitmap (brush) classes
    semantic_classes_json = []
    for obj_class in project_meta.obj_classes:
        obj_class: sly.ObjClass
        if obj_class.geometry_type in [sly.Polygon, sly.Bitmap]:
            semantic_classes_json.append(obj_class.to_json())

    for obj_class in semantic_classes_json:
        obj_class["imagesCount"] = class_images[obj_class["title"]]
        obj_class["objectsCount"] = class_objects[obj_class["title"]]
        obj_class["areaPercent"] = class_area[obj_class["title"]]

    unlabeled_count = 0
    for ds_counter in stats["images"]["datasets"]:
        unlabeled_count += ds_counter["imagesNotMarked"]

    data["classes"] = semantic_classes_json
    state["selectedClasses"] = []
    state["classes"] = len(semantic_classes_json) * [True]
    data["unlabeledCount"] = unlabeled_count

    data["done3"] = False
    state["collapsed3"] = True
    state["disabled3"] = True