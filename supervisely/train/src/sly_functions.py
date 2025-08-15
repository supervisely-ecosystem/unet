import supervisely as sly


def get_bg_class_name(class_names):
    possible_bg_names = ["background", "bg", "unlabeled", "neutral", "__bg__"]
    for name in class_names:
        if name.lower() in possible_bg_names:
            return name
    return None


def get_eval_results_dir_name(api, task_id, project_info):
    task_info = api.task.get_info_by_id(task_id)
    task_dir = f"{task_id}_{task_info['meta']['app']['name']}"
    eval_res_dir = f"/model-benchmark/{project_info.id}_{project_info.name}/{task_dir}/"
    eval_res_dir = api.file.get_free_dir_name(sly.env.team_id(), eval_res_dir)
    return eval_res_dir
