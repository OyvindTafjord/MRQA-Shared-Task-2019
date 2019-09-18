local train_size = 433776;
local batch_size = 2;
local grad_accumulate = 16;
local num_epochs = 2;
local learning_rate = 1e-5;
local cuda_device = 1;
local sample_size_train = 75000;
local sample_size_dev = 1000;
local data_dir = "/home/oyvindt/data/MRQA/";
local bert_model_vocab = "bert-large-cased";
local bert_model = "/home/oyvindt/models/bert_large_cased_whole_word_masking_flat.tar.gz";

{
    "dataset_reader": {
        "type": "mrqa_reader",
        "bert_vocab": bert_model_vocab,
        "is_training":true,
        "sample_size": sample_size_train,
        "token_indexers": {
            "bert": {
                  "type": "bert-pretrained",
                  "pretrained_model": bert_model_vocab,
                  "do_lowercase": false,
                  "use_starting_offsets": true
              }
        }
    },
    "validation_dataset_reader": {
        "type": "mrqa_reader",
        "bert_vocab": bert_model_vocab,
        "lazy": true,
        "sample_size": sample_size_dev,
        "token_indexers": {
            "bert": {
                  "type": "bert-pretrained",
                  "pretrained_model": bert_model_vocab,
                  "do_lowercase": false,
                  "use_starting_offsets": true
              }
        }
    },
    "train_data_path": data_dir+"train/HotpotQA.jsonl.gz,"+data_dir+"train/NaturalQuestions.jsonl.gz,"+data_dir+"train/NewsQA.jsonl.gz,"+data_dir+"train/SearchQA.jsonl.gz,"+data_dir+"train/SQuAD.jsonl.gz,"+data_dir+"train/TriviaQA.jsonl.gz",
    "validation_data_path": data_dir+"in_domain_dev/HotpotQA.jsonl.gz,"+data_dir+"in_domain_dev/NaturalQuestions.jsonl.gz,"+data_dir+"in_domain_dev/NewsQA.jsonl.gz,"+data_dir+"in_domain_dev/SearchQA.jsonl.gz,"+data_dir+"in_domain_dev/SQuAD.jsonl.gz,"+data_dir+"in_domain_dev/TriviaQA.jsonl.gz,"+data_dir+"out_of_domain_dev/BioASQ.jsonl.gz,"+data_dir+"out_of_domain_dev/DROP.jsonl.gz,"+data_dir+"out_of_domain_dev/DuoRC.jsonl.gz,"+data_dir+"out_of_domain_dev/RACE.jsonl.gz,"+data_dir+"out_of_domain_dev/RelationExtraction.jsonl.gz,"+data_dir+"out_of_domain_dev/TextbookQA.jsonl.gz",    
    "evaluate_custom": {
        "metadata_fields": "qid,best_span_str"
    },
    "iterator": {
        "type": "mrqa_iterator",
        "batch_size": batch_size,
        "max_instances_in_memory": 1000
    },
    "model": {
        "type": "BERT_QA",
        "initializer": [],
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model,
                    "requires_grad":true
                }
            }
        }
    },
  "trainer": {
    "optimizer": {
      "type": "bert_adam",
      "weight_decay": 0.01,
      "parameter_groups": [[["bias", "LayerNorm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": 0.1,
      "num_steps_per_epoch": std.ceil(train_size / batch_size),
    },
    "validation_metric": "+f1",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    "grad_accumulate_epochs": grad_accumulate,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device
  },
    "validation_iterator": {
        "type": "mrqa_iterator",
        "batch_size": 6
    }
}
