from typing import Dict, Optional

import datasets

from efficiency_benchmark.task import (BINARY_CLASSIFICATION_METRICS,
                                       ENTAILMENT_METRICS, MT_METRICS,
                                       PERPLEXITY_METRICS, QA_METRICS,
                                       InstanceFormat, Task,
                                       classification_metrics, mc_metrics)
from efficiency_benchmark.tasks.efficiency_benchmark import (
    EfficiencyBenchmarkEleutherClassificationTask,
    EfficiencyBenchmarkEleutherClassificationTaskWithRenamedSplits,
    EfficiencyBenchmarkEleutherTask,
    EfficiencyBenchmarkEleutherTaskWithRenamedSplits,
    EfficiencyBenchmarkHFDatasetsTask, EfficiencyBenchmarkMetaICLTask,
    EfficiencyBenchmarkMrqaTask, EfficiencyBenchmarkPromptTask,
    EfficiencyBenchmarkRaceEleutherTask, EfficiencyBenchmarkRaftTask,
    EfficiencyBenchmarkTranslationTask, EfficiencyBenchmarkHuggingfaceTask, EfficiencyBenchmarkP3Task)
from efficiency_benchmark.tasks.huggingface import (
    hfclassification_conversion, hfmc_conversion, hfqa_conversion)
from efficiency_benchmark.tasks.t5 import t5_prompt_conversion

TASKS: Dict[str, Task] = {
    "huggingface": EfficiencyBenchmarkHuggingfaceTask,
    "wmt16-en-ro": EfficiencyBenchmarkTranslationTask("wmt16", "ro-en").add_metrics(MT_METRICS),
    "wmt16-ro-en": EfficiencyBenchmarkTranslationTask("wmt16", "ro-en").add_metrics(MT_METRICS),
    "wmt14-de-en": EfficiencyBenchmarkTranslationTask("wmt14", "de-en").add_metrics(MT_METRICS),
    "wmt14-en-de": EfficiencyBenchmarkTranslationTask("wmt14", "de-en").add_metrics(MT_METRICS), 

    "wikitext-prompt": EfficiencyBenchmarkPromptTask("wikitext", "wikitext-103-raw-v1"),
    # RAFT
    "raft::ade_corpus_v2": EfficiencyBenchmarkRaftTask("ade_corpus_v2"),
    "raft::banking_77": EfficiencyBenchmarkRaftTask("banking_77"),
    "raft::neurips_impact_statement_risks": EfficiencyBenchmarkRaftTask("neurips_impact_statement_risks"),
    "raft::one_stop_english": EfficiencyBenchmarkRaftTask("one_stop_english"),
    "raft::overruling": EfficiencyBenchmarkRaftTask("overruling"),
    "raft::semiconductor_org_types": EfficiencyBenchmarkRaftTask("semiconductor_org_types"),
    "raft::systematic_review_inclusion": EfficiencyBenchmarkRaftTask("systematic_review_inclusion"),
    "raft::tai_safety_research": EfficiencyBenchmarkRaftTask("tai_safety_research"),
    "raft::terms_of_service": EfficiencyBenchmarkRaftTask("terms_of_service"),
    "raft::tweet_eval_hate": EfficiencyBenchmarkRaftTask("tweet_eval_hate"),
    "raft::twitter_complaints": EfficiencyBenchmarkRaftTask("twitter_complaints"),
    # MRQA
    "mrqa::race": EfficiencyBenchmarkMrqaTask("mrqa", "race").add_metrics(QA_METRICS),
    "mrqa::newsqa": EfficiencyBenchmarkMrqaTask("mrqa", "newsqa").add_metrics(QA_METRICS),
    "mrqa::triviaqa": EfficiencyBenchmarkMrqaTask("mrqa", "triviaqa-web").add_metrics(QA_METRICS),
    "mrqa::searchqa": EfficiencyBenchmarkMrqaTask("mrqa", "searchqa").add_metrics(QA_METRICS),
    "mrqa::hotpotqa": EfficiencyBenchmarkMrqaTask("mrqa", "hotpotqa").add_metrics(QA_METRICS),
    "mrqa::naturalquestions": EfficiencyBenchmarkMrqaTask("mrqa", "naturalquestionsshort").add_metrics(QA_METRICS),
    "mrqa::bioasq": EfficiencyBenchmarkMrqaTask("mrqa", "bioasq").add_metrics(QA_METRICS),
    "mrqa::drop": EfficiencyBenchmarkMrqaTask("mrqa", "drop").add_metrics(QA_METRICS),
    "mrqa::relationextraction": EfficiencyBenchmarkMrqaTask("mrqa", "relationextraction").add_metrics(QA_METRICS),
    "mrqa::textbookqa": EfficiencyBenchmarkMrqaTask("mrqa", "textbookqa").add_metrics(QA_METRICS),
    "mrqa::duorc.paraphraserc": EfficiencyBenchmarkMrqaTask("mrqa", "duorc.paraphraserc").add_metrics(QA_METRICS),

    # MetaICL
    "metaicl::piqa": EfficiencyBenchmarkMetaICLTask("piqa").add_metrics(mc_metrics(2)),
    "metaicl::boolq": EfficiencyBenchmarkMetaICLTask("boolq").add_metrics(classification_metrics(2)),

    "metaicl::tweet_eval-stance_feminist": EfficiencyBenchmarkMetaICLTask("tweet_eval-stance_feminist").add_metrics(classification_metrics(3)),
    "metaicl::ethos-national_origin": EfficiencyBenchmarkMetaICLTask("ethos-national_origin").add_metrics(classification_metrics(2)),
    "metaicl::tweet_eval-hate": EfficiencyBenchmarkMetaICLTask("tweet_eval-hate").add_metrics(classification_metrics(2)),
    "metaicl::ag_news": EfficiencyBenchmarkMetaICLTask("ag_news").add_metrics(classification_metrics(4)),
    "metaicl::amazon_polarity": EfficiencyBenchmarkMetaICLTask("amazon_polarity").add_metrics(classification_metrics(2)),
    "metaicl::hate_speech18": EfficiencyBenchmarkMetaICLTask("hate_speech18").add_metrics(classification_metrics(2)),
    "metaicl::poem_sentiment": EfficiencyBenchmarkMetaICLTask("poem_sentiment").add_metrics(classification_metrics(3)),
    "metaicl::climate_fever": EfficiencyBenchmarkMetaICLTask("climate_fever").add_metrics(classification_metrics(4)),
    "metaicl::medical_questions_pairs": EfficiencyBenchmarkMetaICLTask("medical_questions_pairs").add_metrics(classification_metrics(2)),
    "metaicl::tweet_eval-stance_atheism": EfficiencyBenchmarkMetaICLTask("tweet_eval-stance_atheism").add_metrics(classification_metrics(3)),
    "metaicl::superglue-cb": EfficiencyBenchmarkMetaICLTask("superglue-cb").add_metrics(classification_metrics(3)),
    "metaicl::dbpedia_14": EfficiencyBenchmarkMetaICLTask("dbpedia_14").add_metrics(classification_metrics(14)),
    "metaicl::wiki_qa": EfficiencyBenchmarkMetaICLTask("wiki_qa").add_metrics(classification_metrics(2)),
    "metaicl::emo": EfficiencyBenchmarkMetaICLTask("emo").add_metrics(classification_metrics(4)),
    "metaicl::yelp_polarity": EfficiencyBenchmarkMetaICLTask("yelp_polarity").add_metrics(classification_metrics(2)),
    "metaicl::ethos-religion": EfficiencyBenchmarkMetaICLTask("ethos-religion").add_metrics(classification_metrics(2)),
    "metaicl::financial_phrasebank": EfficiencyBenchmarkMetaICLTask("financial_phrasebank").add_metrics(classification_metrics(3)),
    "metaicl::tab_fact": EfficiencyBenchmarkMetaICLTask("tab_fact").add_metrics(classification_metrics(2)),
    "metaicl::anli": EfficiencyBenchmarkMetaICLTask("anli").add_metrics(classification_metrics(3)),
    "metaicl::ethos-race": EfficiencyBenchmarkMetaICLTask("ethos-race").add_metrics(classification_metrics(2)),

    "metaicl::glue-mrpc": EfficiencyBenchmarkMetaICLTask("glue-mrpc").add_metrics(classification_metrics(2)),
    "metaicl::glue-qqp": EfficiencyBenchmarkMetaICLTask("glue-qqp").add_metrics(classification_metrics(2)),
    # "metaicl::medical_questions_pairs": EfficiencyBenchmarkMetaICLTask("medical_questions_pairs").add_metrics(classification_metrics(2)),
    "metaicl::paws": EfficiencyBenchmarkMetaICLTask("paws").add_metrics(classification_metrics(2)),

    # "metaicl::anli": EfficiencyBenchmarkMetaICLTask("anli").add_metrics(classification_metrics(3)),
    "metaicl::glue-mnli": EfficiencyBenchmarkMetaICLTask("glue-mnli").add_metrics(classification_metrics(3)),
    "metaicl::glue-qnli": EfficiencyBenchmarkMetaICLTask("glue-qnli").add_metrics(classification_metrics(2)),
    "metaicl::glue-rte": EfficiencyBenchmarkMetaICLTask("glue-rte").add_metrics(classification_metrics(2)),
    "metaicl::glue-wnli": EfficiencyBenchmarkMetaICLTask("glue-wnli").add_metrics(classification_metrics(2)),
    "metaicl::scitail": EfficiencyBenchmarkMetaICLTask("scitail").add_metrics(classification_metrics(2)),
    "metaicl::sick": EfficiencyBenchmarkMetaICLTask("sick").add_metrics(classification_metrics(3)),
    # "metaicl::superglue-cb": EfficiencyBenchmarkMetaICLTask("superglue-cb").add_metrics(classification_metrics(3)),

    "metaicl::ai2_arc": EfficiencyBenchmarkMetaICLTask("ai2_arc").add_metrics(mc_metrics(4)),
    "metaicl::codah": EfficiencyBenchmarkMetaICLTask("codah").add_metrics(mc_metrics(4)),
    "metaicl::cosmos_qa": EfficiencyBenchmarkMetaICLTask("cosmos_qa").add_metrics(mc_metrics(4)),
    "metaicl::dream": EfficiencyBenchmarkMetaICLTask("dream").add_metrics(mc_metrics(3)),
    "metaicl::hellaswag": EfficiencyBenchmarkMetaICLTask("hellaswag").add_metrics(mc_metrics(4)),
    "metaicl::openbookqa": EfficiencyBenchmarkMetaICLTask("openbookqa").add_metrics(mc_metrics(4)),
    "metaicl::qasc": EfficiencyBenchmarkMetaICLTask("qasc").add_metrics(mc_metrics(8)),
    "metaicl::quail": EfficiencyBenchmarkMetaICLTask("quail").add_metrics(mc_metrics(4)),
    "metaicl::quarel": EfficiencyBenchmarkMetaICLTask("quarel").add_metrics(mc_metrics(2)),
    "metaicl::quartz-no_knowledge": EfficiencyBenchmarkMetaICLTask("quartz-no_knowledge").add_metrics(mc_metrics(2)),
    "metaicl::quartz-with_knowledge": EfficiencyBenchmarkMetaICLTask("quartz-with_knowledge").add_metrics(mc_metrics(2)),
    "metaicl::sciq": EfficiencyBenchmarkMetaICLTask("sciq").add_metrics(mc_metrics(4)),
    "metaicl::superglue-copa": EfficiencyBenchmarkMetaICLTask("superglue-copa").add_metrics(mc_metrics(2)),
    "metaicl::swag": EfficiencyBenchmarkMetaICLTask("swag").add_metrics(mc_metrics(4)),
    "metaicl::wino_grande": EfficiencyBenchmarkMetaICLTask("wino_grande").add_metrics(mc_metrics(2)),
    "metaicl::wiqa": EfficiencyBenchmarkMetaICLTask("wiqa").add_metrics(mc_metrics(3)),
    "metaicl::unifiedqa:qasc": EfficiencyBenchmarkMetaICLTask("unifiedqa:qasc").add_metrics(mc_metrics(8)),
    "metaicl::unifiedqa:qasc_with_ir": EfficiencyBenchmarkMetaICLTask("unifiedqa:qasc_with_ir").add_metrics(mc_metrics(8)),
    "metaicl::unifiedqa:openbookqa": EfficiencyBenchmarkMetaICLTask("unifiedqa:openbookqa").add_metrics(mc_metrics(4)),
    "metaicl::unifiedqa:openbookqa_with_ir": EfficiencyBenchmarkMetaICLTask("unifiedqa:openbookqa_with_ir").add_metrics(mc_metrics(4)),
    "metaicl::unifiedqa:mctest": EfficiencyBenchmarkMetaICLTask("unifiedqa:mctest").add_metrics(mc_metrics(4)),
    "metaicl::unifiedqa:ai2_science_middle": EfficiencyBenchmarkMetaICLTask("unifiedqa:ai2_science_middle").add_metrics(mc_metrics(4)),
    "metaicl::numer_sense": EfficiencyBenchmarkMetaICLTask("numer_sense").add_metrics(classification_metrics(12)),
    "metaicl::race-high": EfficiencyBenchmarkMetaICLTask("race-high").add_metrics(mc_metrics(4)),
    "metaicl::commonsense_qa": EfficiencyBenchmarkMetaICLTask("commonsense_qa").add_metrics(mc_metrics(5)),

    "piqa": EfficiencyBenchmarkEleutherTask("piqa").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="goal",
            answer_choices_fields=["sol1", "sol2"],
            correct_answer_index_field="label"
        )
    ).add_metrics(mc_metrics(2)),
    "squad": EfficiencyBenchmarkHFDatasetsTask("squad").add_instance_conversion(
        InstanceFormat.HF_QA,
        hfqa_conversion()
    ).add_metrics(QA_METRICS),
    "squadshifts-reddit": EfficiencyBenchmarkHFDatasetsTask("squadshifts", "reddit").add_instance_conversion(
        InstanceFormat.HF_QA,
        hfqa_conversion()   
    ).add_metrics(QA_METRICS),
    "squadshifts-amazon": EfficiencyBenchmarkHFDatasetsTask("squadshifts", "amazon").add_instance_conversion(
        InstanceFormat.HF_QA,
        hfqa_conversion()   
    ).add_metrics(QA_METRICS),
    "squadshifts-nyt": EfficiencyBenchmarkHFDatasetsTask("squadshifts", "nyt").add_instance_conversion(
        InstanceFormat.HF_QA,
        hfqa_conversion()   
    ).add_metrics(QA_METRICS),
    "squadshifts-new-wiki": EfficiencyBenchmarkHFDatasetsTask("squadshifts", "new_wiki").add_instance_conversion(
        InstanceFormat.HF_QA,
        hfqa_conversion()   
    ).add_metrics(QA_METRICS),
    "squad2": EfficiencyBenchmarkEleutherTask("squad2").add_metrics(QA_METRICS),
    "rte": EfficiencyBenchmarkEleutherClassificationTask(
        "rte",
        answer_options=["True", "False"]
    ).add_instance_conversion(
        InstanceFormat.T5_PROMPT,
        t5_prompt_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            use_fields=["sentence1", "sentence2"]
        )
    ).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            premise_field="sentence1",
            hypothesis_field="sentence2"
        )
    ),
    "superglue::rte": EfficiencyBenchmarkHFDatasetsTask("super_glue", "rte").add_instance_conversion(
        InstanceFormat.T5_PROMPT,
        t5_prompt_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            use_fields=["premise", "hypothesis"]
        )
    ).add_metrics(ENTAILMENT_METRICS),
    "cola": EfficiencyBenchmarkEleutherClassificationTask("cola", answer_options=["no", "yes"]).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="cola",
            label_map={0: "unacceptable", 1: "acceptable"},
            premise_field="sentence",
            hypothesis_field=None,
            id_field='idx'
        )
    ),
    "mnli": EfficiencyBenchmarkEleutherClassificationTaskWithRenamedSplits(
        "mnli",
        answer_options=["True", "Neither", "False"]
    ).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="mnli",
            label_map={0: "entailment", 1: "neutral", 2: "contradiction"},
            id_field='idx'
        )
    ),
    "mnli_mismatched": EfficiencyBenchmarkEleutherClassificationTask(
        "mnli_mismatched",
        answer_options=["True", "Neither", "False"]
    ).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="mnli",
            label_map={0: "entailment", 1: "neutral", 2: "contradiction"},
            id_field='idx')
    ),
    "mrpc": EfficiencyBenchmarkEleutherClassificationTask("mrpc", answer_options=["no", "yes"]).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="mrpc",
            label_map={0: "not_equivalent", 1: "equivalent"},
            premise_field="sentence1",
            hypothesis_field="sentence2",
            id_field='idx'
        )
    ),
    "qnli": EfficiencyBenchmarkEleutherClassificationTask("qnli", answer_options=["yes", "no"]).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="qnli",
            label_map={0: "entailment", 1: "not_entailment"},
            premise_field="question",
            hypothesis_field="sentence",
            id_field='idx'
        )
    ),
    "qqp": EfficiencyBenchmarkEleutherClassificationTask("qqp", answer_options=["no", "yes"]).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="qqp",
            label_map={0: "not_duplicate", 1: "duplicate"},
            premise_field="question1",
            hypothesis_field="question2",
            id_field='idx'
        )
    ),
    "sst": EfficiencyBenchmarkEleutherClassificationTask("sst", answer_options=["negative", "positive"]).add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="sst",
            label_map={0: "negative", 1: "positive"},
            premise_field="sentence",
            hypothesis_field=None,
            id_field='idx'
        )
    ),
    
    "wnli": EfficiencyBenchmarkEleutherTask("wnli").add_metrics(ENTAILMENT_METRICS),
    "boolq": EfficiencyBenchmarkEleutherTask("boolq").add_metrics(classification_metrics(2)),
    "cb": EfficiencyBenchmarkEleutherTask("cb").add_metrics(ENTAILMENT_METRICS),
    "copa": EfficiencyBenchmarkEleutherTask("copa").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="premise",
            answer_choices_fields=["choice1", "choice2"],
            correct_answer_index_field="label",
            id_field="idx"
        )
    ).add_metrics(mc_metrics(2)),
    "multirc": EfficiencyBenchmarkEleutherTask("multirc").add_metrics(QA_METRICS),
    #"record": EfficiencyBenchmarkEleutherTask("record"),    # record doesn't have a 1:1 correspondence between HF instances and EAI instances
    "wic": EfficiencyBenchmarkEleutherTask("wic").add_metrics(ENTAILMENT_METRICS),
    "wsc": EfficiencyBenchmarkEleutherTask("wsc").add_metrics(mc_metrics(2)),
    #"coqa": EfficiencyBenchmarkEleutherTask("coqa"),  # currently broken in the datasets library
    "drop": EfficiencyBenchmarkEleutherTask("drop").add_metrics(QA_METRICS),
    "lambada": EfficiencyBenchmarkEleutherTask("lambada_standard"),
    "lambada_cloze": EfficiencyBenchmarkEleutherTask("lambada_standard_cloze"),
    "lambada_mt_en": EfficiencyBenchmarkEleutherTask("lambada_openai_mt_en"),
    "lambada_mt_fr": EfficiencyBenchmarkEleutherTask("lambada_openai_mt_fr"),
    "lambada_mt_de": EfficiencyBenchmarkEleutherTask("lambada_openai_mt_de"),
    "lambada_mt_it": EfficiencyBenchmarkEleutherTask("lambada_openai_mt_it"),
    "lambada_mt_es": EfficiencyBenchmarkEleutherTask("lambada_openai_mt_es"),
    "prost": EfficiencyBenchmarkEleutherTask("prost").add_metrics(mc_metrics(4)),
    "mc_taco": EfficiencyBenchmarkEleutherTask("mc_taco").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "pubmedqa": EfficiencyBenchmarkEleutherTaskWithRenamedSplits("pubmedqa").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "sciq": EfficiencyBenchmarkEleutherTask("sciq").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question",
            answer_choices_fields=["correct_answer", "distractor1", "distractor2", "distractor3"],
            correct_answer_field="correct_answer"
        )
    ).add_metrics(mc_metrics(4)),
    "qa4mre_2011": EfficiencyBenchmarkEleutherTask("qa4mre_2011").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="document_str",
            question_field="question_str",
            answer_choices_fields="answer_options.answer_str",
            correct_answer_index_field="correct_answer_id",
            answer_mappings={'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        )
    ).add_metrics(mc_metrics(5)),
    "qa4mre_2012": EfficiencyBenchmarkEleutherTask("qa4mre_2012").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="document_str",
            question_field="question_str",
            answer_choices_fields="answer_options.answer_str",
            correct_answer_index_field="correct_answer_id",
            answer_mappings={'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        )
    ).add_metrics(mc_metrics(5)),
    "qa4mre_2013": EfficiencyBenchmarkEleutherTask("qa4mre_2013").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="document_str",
            question_field="question_str",
            answer_choices_fields="answer_options.answer_str",
            correct_answer_index_field="correct_answer_id",
            answer_mappings={'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        )
    ).add_metrics(mc_metrics(5)),
    "triviaqa": EfficiencyBenchmarkEleutherTask(
        "triviaqa"
    ).add_metrics(QA_METRICS),
    "arc_easy": EfficiencyBenchmarkEleutherTask("arc_easy").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question",
            answer_choices_fields="choices.text",
            correct_answer_index_field="answerKey",
            id_field="id",
            answer_mappings={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3}
        )
    ).add_metrics(mc_metrics(4)),
    "arc_challenge": EfficiencyBenchmarkEleutherTask("arc_challenge").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question",
            answer_choices_fields="choices.text",
            correct_answer_index_field="answerKey",
            id_field="id",
            answer_mappings={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3}
        )
    ).add_metrics(mc_metrics(4)),
    "logiqa": EfficiencyBenchmarkEleutherTask("logiqa").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="context",
            question_field="question",
            answer_choices_fields="options",
            correct_answer_index_field="label"
        )
    ).add_metrics(mc_metrics(4)),
    "hellaswag": EfficiencyBenchmarkEleutherTask("hellaswag").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="ctx",
            answer_choices_fields="endings",
            correct_answer_index_field="label",
            answer_mappings={'0': 0, '1': 1, '2': 2, '3': 3}
        )
    ).add_metrics(mc_metrics(4)),
    "openbookqa": EfficiencyBenchmarkEleutherTask("openbookqa").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question_stem",
            answer_choices_fields="choices.text",
            correct_answer_index_field="answerKey",
            id_field="id"
        )
    ).add_metrics(mc_metrics(4)),
    "race": EfficiencyBenchmarkHFDatasetsTask("race", "high").add_metrics(mc_metrics(4)),
    "eleuther::race": EfficiencyBenchmarkRaceEleutherTask().add_metrics(mc_metrics(4)),
    "headqa_es": EfficiencyBenchmarkEleutherTask("headqa_es").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="qtext",
            answer_choices_fields=[
                "answers.0.atext",
                "answers.1.atext",
                "answers.2.atext",
                "answers.3.atext",
                "answers.4.atext"
            ],
            correct_answer_index_field="ra",
            answer_mappings={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        )
    ).add_metrics(mc_metrics(5)),
    "headqa_en": EfficiencyBenchmarkEleutherTask("headqa_en").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="qtext",
            answer_choices_fields=[
                "answers.0.atext",
                "answers.1.atext",
                "answers.2.atext",
                "answers.3.atext",
                "answers.4.atext"
            ],
            correct_answer_index_field="ra",
            answer_mappings={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        )
    ).add_metrics(mc_metrics(5)),
    "mathqa": EfficiencyBenchmarkEleutherTask("mathqa").add_metrics(mc_metrics(5)),
    "webqs": EfficiencyBenchmarkEleutherTask("webqs").add_metrics(QA_METRICS),
    "wsc273": EfficiencyBenchmarkEleutherTask("wsc273").add_metrics(ENTAILMENT_METRICS),
    "winogrande": EfficiencyBenchmarkEleutherTask("winogrande").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="sentence",
            answer_choices_fields=["option1", "option2"],
            correct_answer_index_field="answer",
            answer_mappings={'1': 0, '2': 1}
        )
    ).add_metrics(mc_metrics(2)),
    "anli_r1": EfficiencyBenchmarkEleutherTask("anli_r1").add_metrics(ENTAILMENT_METRICS),
    "anli_r2": EfficiencyBenchmarkEleutherTask("anli_r2").add_metrics(ENTAILMENT_METRICS),
    "anli_r3": EfficiencyBenchmarkEleutherTask("anli_r3").add_metrics(ENTAILMENT_METRICS),
    "ethics_cm": EfficiencyBenchmarkEleutherTask("ethics_cm").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_deontology": EfficiencyBenchmarkEleutherTask("ethics_deontology").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_justice": EfficiencyBenchmarkEleutherTask("ethics_justice").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_utilitarianism_original": EfficiencyBenchmarkEleutherTask("ethics_utilitarianism_original").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_utilitarianism": EfficiencyBenchmarkEleutherTask("ethics_utilitarianism").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_virtue": EfficiencyBenchmarkEleutherTask("ethics_virtue").add_metrics(BINARY_CLASSIFICATION_METRICS),
    # "truthfulqa_mc": EfficiencyBenchmarkEleutherTask("truthfulqa_mc"),
    "truthfulqa_gen": EfficiencyBenchmarkEleutherTask("truthfulqa_gen"),
    "mutual": EfficiencyBenchmarkEleutherTask("mutual"),
    "mutual_plus": EfficiencyBenchmarkEleutherTask("mutual_plus"),
    "math_algebra": EfficiencyBenchmarkEleutherTask("math_algebra").add_metrics(QA_METRICS),
    "math_counting_and_prob": EfficiencyBenchmarkEleutherTask("math_counting_and_prob").add_metrics(QA_METRICS),
    "math_geometry": EfficiencyBenchmarkEleutherTask("math_geometry").add_metrics(QA_METRICS),
    "math_intermediate_algebra": EfficiencyBenchmarkEleutherTask("math_intermediate_algebra").add_metrics(QA_METRICS),
    "math_num_theory": EfficiencyBenchmarkEleutherTask("math_num_theory").add_metrics(QA_METRICS),
    "math_prealgebra": EfficiencyBenchmarkEleutherTask("math_prealgebra").add_metrics(QA_METRICS),
    "math_precalc": EfficiencyBenchmarkEleutherTask("math_precalc").add_metrics(QA_METRICS),
    "math_asdiv": EfficiencyBenchmarkEleutherTask("math_asdiv").add_metrics(QA_METRICS),
    "arithmetic_2da": EfficiencyBenchmarkEleutherTask("arithmetic_2da").add_metrics(QA_METRICS),
    "arithmetic_2ds": EfficiencyBenchmarkEleutherTask("arithmetic_2ds").add_metrics(QA_METRICS),
    "arithmetic_3da": EfficiencyBenchmarkEleutherTask("arithmetic_3da").add_metrics(QA_METRICS),
    "arithmetic_3ds": EfficiencyBenchmarkEleutherTask("arithmetic_3ds").add_metrics(QA_METRICS),
    "arithmetic_4da": EfficiencyBenchmarkEleutherTask("arithmetic_4da").add_metrics(QA_METRICS),
    "arithmetic_4ds": EfficiencyBenchmarkEleutherTask("arithmetic_4ds").add_metrics(QA_METRICS),
    "arithmetic_5da": EfficiencyBenchmarkEleutherTask("arithmetic_5da").add_metrics(QA_METRICS),
    "arithmetic_5ds": EfficiencyBenchmarkEleutherTask("arithmetic_5ds").add_metrics(QA_METRICS),
    "arithmetic_2dm": EfficiencyBenchmarkEleutherTask("arithmetic_2dm").add_metrics(QA_METRICS),
    "arithmetic_1dc": EfficiencyBenchmarkEleutherTask("arithmetic_1dc").add_metrics(QA_METRICS),
    "anagrams1": EfficiencyBenchmarkEleutherTask("anagrams1").add_metrics(QA_METRICS),
    "anagrams2": EfficiencyBenchmarkEleutherTask("anagrams2").add_metrics(QA_METRICS),
    "cycle_letters": EfficiencyBenchmarkEleutherTask("cycle_letters").add_metrics(QA_METRICS),
    "random_insertion": EfficiencyBenchmarkEleutherTask("random_insertion").add_metrics(QA_METRICS),
    "reversed_words": EfficiencyBenchmarkEleutherTask("reversed_words").add_metrics(QA_METRICS),

    # from catwalk
    "wikitext": EfficiencyBenchmarkEleutherTask("wikitext").add_metrics(PERPLEXITY_METRICS),
}

for config in datasets.get_dataset_config_names("bigscience/P3"):
    TASKS[f"p3::{config}"] = EfficiencyBenchmarkP3Task(config)

TASK_SETS = {
    "iz": {
        "arc_challenge",
        "arc_easy",
        "boolq",
        "copa",
        "headqa_en",
        "hellaswag",
        "lambada",
        "logiqa",
        "mathqa",
        "mc_taco",
        "mrpc",
        "multirc",
        "openbookqa",
        "piqa",
        "prost",
        "pubmedqa",
        "qnli",
        "qqp",
        "race",
        "rte",
        "sciq",
        "sst",
        "triviaqa",
        "webqs",
        "wic",
        "winogrande",
        "wnli",
        "wsc",
    },
    "raft": {name for name in TASKS.keys() if name.startswith("raft::")},
    "metaicl-classification-eval": {
        "metaicl::tweet_eval-stance_feminist",
        "metaicl::ethos-national_origin",
        "metaicl::tweet_eval-hate",
        "metaicl::ag_news",
        "metaicl::amazon_polarity",
        "metaicl::hate_speech18",
        "metaicl::poem_sentiment",
        "metaicl::climate_fever",
        "metaicl::medical_questions_pairs",
        "metaicl::tweet_eval-stance_atheism",
        "metaicl::superglue-cb",
        "metaicl::dbpedia_14",
        "metaicl::wiki_qa",
        "metaicl::emo",
        "metaicl::yelp_polarity",
        "metaicl::ethos-religion",
        "metaicl::financial_phrasebank",
        "metaicl::tab_fact",
        "metaicl::anli",
        "metaicl::ethos-race"
    },
    "metaicl-paraphrase-eval": {
        "metaicl::glue-mrpc",
        "metaicl::glue-qqp",
        "metaicl::medical_questions_pairs",
        "metaicl::paws"
    },
    "metaicl-nli-eval": {
        "metaicl::anli",
        "metaicl::glue-mnli",
        "metaicl::glue-qnli",
        "metaicl::glue-rte",
        "metaicl::glue-wnli",
        "metaicl::scitail",
        "metaicl::sick",
        "metaicl::superglue-cb"
    },
    "metaicl-qa-eval": {
        "metaicl::ai2_arc",
        "metaicl::codah",
        "metaicl::cosmos_qa",
        "metaicl::dream",
        "metaicl::hellaswag",
        "metaicl::openbookqa",
        "metaicl::qasc",
        "metaicl::quail",
        "metaicl::quarel",
        "metaicl::quartz-no_knowledge",
        "metaicl::quartz-with_knowledge",
        "metaicl::sciq",
        "metaicl::superglue-copa",
        "metaicl::swag",
        "metaicl::wino_grande",
        "metaicl::wiqa",
        "metaicl::unifiedqa:qasc",
        "metaicl::unifiedqa:qasc_with_ir",
        "metaicl::unifiedqa:openbookqa",
        "metaicl::unifiedqa:openbookqa_with_ir",
        "metaicl::unifiedqa:mctest",
        "metaicl::unifiedqa:ai2_science_middle"
    },
    "metaicl-lr-eval": {
        "metaicl::quarel",
        "metaicl::financial_phrasebank",
        "metaicl::openbookqa",
        "metaicl::codah",
        "metaicl::qasc",
        "metaicl::glue-mrpc",
        "metaicl::dream",
        "metaicl::sick",
        "metaicl::commonsense_qa",
        "metaicl::medical_questions_pairs",
        "metaicl::quartz-with_knowledge",
        "metaicl::poem_sentiment",
        "metaicl::quartz-no_knowledge",
        "metaicl::glue-wnli",
        "metaicl::climate_fever",
        "metaicl::ethos-national_origin",
        "metaicl::ethos-race",
        "metaicl::ethos-religion",
        "metaicl::ai2_arc",
        "metaicl::hate_speech18",
        "metaicl::glue-rte",
        "metaicl::superglue-cb",
        "metaicl::superglue-copa",
        "metaicl::tweet_eval-hate",
        "metaicl::tweet_eval-stance_atheism",
        "metaicl::tweet_eval-stance_feminist"
    }
}


def short_name_for_task_object(task: Task) -> Optional[str]:
    for task_name, task_object in TASKS.items():
        if id(task) == id(task_object):
            return task_name
    return None
