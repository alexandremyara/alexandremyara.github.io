import datasets
import os
import json
import glob

DATA_PATH = "data"
class frenchSlideVQA(datasets.GeneratorBasedBuilder) :
    VERSION = datasets.Version("0.0.1")
    BUILDER_CONFIGS = [ datasets.BuilderConfig(name="complet", version=VERSION, description=""),
                        datasets.BuilderConfig(name="minimal", version=VERSION, description="")]
    DEFAULT_CONFIG_NAME = "complet"

    def _info(self) :
        if self.config.name == "complet" :
            features = datasets.Features({
                                        "id": datasets.Value("int32"),
                                        "presentation_url": datasets.Value("string"),
                                        "title": datasets.Value("string"),
                                        "author": datasets.Value("string"),
                                        "date": datasets.Value("string"), #AAAA-MM-JJ
                                        "len": datasets.Value("int32"),
                                        "description": datasets.Value("string"),
                                        "lang": datasets.Value("string"),
                                        "dim": datasets.Sequence(datasets.Value("int32")),
                                        "like": datasets.Value("int32"),
                                        "view": datasets.Value("int32"),
                                        "transcript": datasets.Sequence(datasets.Value("string")),
                                        "most_read": datasets.Sequence(datasets.Value("int32")),
                                        "images": datasets.Sequence(datasets.Image()),
                                        "questions/answers": datasets.Sequence({"question": datasets.Value("string"), "answer": datasets.Value("string")})
                                        })
        else :
            features = datasets.Features({
                                        "id": datasets.Value("int32"),
                                        "presentation_url": datasets.Value("string"),
                                        "images": datasets.Sequence(datasets.Image()),
                                        "questions/answers": datasets.Sequence({"question": datasets.Value("string"), "answer": datasets.Value("string")})
                                        })
        return datasets.DatasetInfo(features=features, version=self.VERSION)
    
    def _split_generators(self, dl_manager) :
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": DATA_PATH
                },
            ),
        ]
    
    def _generate_examples(self, data_dir) :
        dirs = [dir for dir in os.listdir(data_dir)]

        for idx, dir in enumerate(dirs) :
            with open(f"{DATA_PATH}/{dir}/{dir}.json", "r") as f:
                meta_data = json.loads(f.read())
            with open(f"{DATA_PATH}/{dir}/qa_{dir}.json", "r") as f :
               qas =  json.loads(f.read())
            
            img_names = glob.glob(f"{DATA_PATH}/{dir}/*.jpg")

            if self.config.name == "complet" :
                yield idx, {"id": int(meta_data["id"]),
                            "presentation_url": meta_data["presentation_url"],
                            "title": meta_data["title"],
                            "author": meta_data["author"],
                            "date": meta_data["date"], #AAAA-MM-JJ
                            "len": meta_data["len"],
                            "description": meta_data["description"],
                            "lang": meta_data["lang"],
                            "dim": list(meta_data["dim"].values()),
                            "like": int(meta_data["like"]),
                            "view": int(meta_data["view"]),
                            "transcript": meta_data["transcript"],
                            "most_read": meta_data["mostRead"],
                            "images": img_names,
                            "questions/answers": qas
                            }
            else :
                yield idx, {"id": int(meta_data["id"]),
                            "presentation_url": meta_data["presentation_url"],
                            "images": img_names,
                            "questions/answers": qas
                            }