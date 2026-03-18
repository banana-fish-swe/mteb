import numpy as np

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader
from mteb.types import PromptType

SGPT_CITATION = """@article{muennighoff2022sgpt,
  title={SGPT: GPT Sentence Embeddings for Semantic Search},
  author={Muennighoff, Niklas},
  journal={arXiv preprint arXiv:2202.08904},
  year={2022}
}"""

# Training datasets
_NLI_DATASETS = {"SNLI", "MultiNLI"}
_MSMARCO_DATASETS = {"MSMARCO"}


class _SGPTSpecBModel:
    """Wrapper for SGPT Bi-Encoder models trained with special brackets (specb).

    Queries are wrapped with [query] and documents with {doc} to signal
    asymmetric search roles, as described in Section 4.2 of the paper.
    """

    def __init__(self, model_name: str, revision: str | None = None, **kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ):
        texts = [text for batch in inputs for text in batch["text"]]
        if prompt_type == PromptType.query:
            texts = ["[" + t + "]" for t in texts]
        else:
            texts = ["{" + t + "}" for t in texts]
        embeddings = self.model.encode(texts, **kwargs)
        if hasattr(embeddings, "cpu"):
            return embeddings.cpu().detach().float().numpy()
        return np.array(embeddings)


def sgpt_specb_loader(model_name: str, revision: str | None = None, **kwargs):
    return _SGPTSpecBModel(model_name, revision=revision, **kwargs)


# ---------------------------------------------------------------------------
# BLOOM-based SGPT model
# ---------------------------------------------------------------------------

bigscience__sgpt_bloom_7b1_msmarco = ModelMeta(
    name="bigscience/sgpt-bloom-7b1-msmarco",
    model_type=["dense"],
    revision="dc579f3d2d5a0795eba2049e16c3e36c74007ad3",
    release_date="2022-08-26",
    languages=None,
    loader=sentence_transformers_loader,
    loader_kwargs={"model_kwargs": {"torch_dtype": "auto"}},
    n_parameters=7_100_000_000,
    n_embedding_parameters=1_026_793_472,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=4096,
    license=None,
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="/gpfsscratch/rech/six/commun/commun/experiments/muennighoff/bloomckpt/6b3/bloom-7b1",
    superseded_by=None,
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 5.8B — GPT-J base, asymmetric MSMARCO + specb
# ---------------------------------------------------------------------------

muennighoff__sgpt_5b8_weightedmean_msmarco_specb_bitfit = ModelMeta(
    name="Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    loader_kwargs={"model_kwargs": {"torch_dtype": "auto"}},
    revision="2dbba11efed19bb418811eac04be241ddc42eb99",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=5_900_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-j-6b",
    superseded_by=None,
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 2.7B — GPT-Neo 2.7B base, asymmetric MSMARCO + specb
# ---------------------------------------------------------------------------

muennighoff__sgpt_2b7_weightedmean_msmarco_specb_bitfit = ModelMeta(
    name="Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    revision="44ec091e09d0366b9681dc853f88bfb619fedd19",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=2_700_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=2560,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-2.7B",
    superseded_by=None,
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 1.3B — GPT-Neo 1.3B base, asymmetric MSMARCO + specb
# ---------------------------------------------------------------------------

muennighoff__sgpt_1b3_weightedmean_msmarco_specb_bitfit = ModelMeta(
    name="Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    revision="bac636ea26b5b810eee6be73fe8c8f484ba1c8c4",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=1_300_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-1.3B",
    superseded_by=None,
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 125M — GPT-Neo 125M base, asymmetric MSMARCO + specb
# ---------------------------------------------------------------------------

muennighoff__sgpt_125m_weightedmean_msmarco_specb_bitfit = ModelMeta(
    name="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    revision="c135b3471857e3d8db7fd00ee981d7d1948f0770",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by=None,
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_weightedmean_msmarco_specb = ModelMeta(
    name="Muennighoff/SGPT-125M-weightedmean-msmarco-specb",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    revision="746b4de85fb43ccd9046fe8526080953f679c94a",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-msmarco-specb",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_lasttoken_msmarco_specb = ModelMeta(
    name="Muennighoff/SGPT-125M-lasttoken-msmarco-specb",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    revision="1bd81d7dcc03ef718fbe6e798989bef7dd485aac",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-lasttoken-msmarco-specb",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_weightedmean_msmarco_specb_bitfitwte = ModelMeta(
    name="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfitwte",
    model_type=["dense"],
    loader=sgpt_specb_loader,
    revision="fb066e972eb4333c2bda0a1ceac4352bb5ac93fb",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfitwte",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 125M — GPT-Neo 125M base, MSMARCO without specb
# ---------------------------------------------------------------------------

muennighoff__sgpt_125m_weightedmean_msmarco = ModelMeta(
    name="Muennighoff/SGPT-125M-weightedmean-msmarco",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="08f7c1354599d32faace4d88f7e1d08eda5c6637",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-msmarco",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_weightedmean_msmarco_asym = ModelMeta(
    name="Muennighoff/SGPT-125M-weightedmean-msmarco-asym",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="c12ade8be484c4016711c41127604417e2f3a938",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=300,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-msmarco-asym",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_MSMARCO_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit",
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 5.8B — GPT-J base, symmetric NLI (BitFit)
# ---------------------------------------------------------------------------

muennighoff__sgpt_5b8_weightedmean_nli_bitfit = ModelMeta(
    name="Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    loader_kwargs={"model_kwargs": {"torch_dtype": "auto"}},
    revision="88aafd983654e7ae20241b1f58223de316f7b003",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=5_900_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=4096,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-j-6b",
    superseded_by=None,
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 2.7B — GPT-Neo 2.7B base, symmetric NLI
# ---------------------------------------------------------------------------

muennighoff__sgpt_2b7_weightedmean_nli_bitfit = ModelMeta(
    name="Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="3f56086f795e8562fe8cb97178f23ed6fa453edb",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=2_700_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=2560,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-2.7B",
    superseded_by=None,
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 1.3B — GPT-Neo 1.3B base, symmetric NLI
# ---------------------------------------------------------------------------

muennighoff__sgpt_1b3_weightedmean_nli_bitfit = ModelMeta(
    name="Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="21ac01bac24bf051aa64428d105d95921ec4e562",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=1_300_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-1.3B",
    superseded_by=None,
    citation=SGPT_CITATION,
)

muennighoff__sgpt_1b3_weightedmean_nli = ModelMeta(
    name="Muennighoff/SGPT-1.3B-weightedmean-nli",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="72a2b83739fcff05da3190fb49ead86866464bd2",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=1_300_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-1.3B-weightedmean-nli",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-1.3B",
    superseded_by="Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_1b3_mean_nli = ModelMeta(
    name="Muennighoff/SGPT-1.3B-mean-nli",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="ca9c84a839fd4f59e6ef70265cd83e9d3af50c01",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=1_300_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=2048,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-1.3B-mean-nli",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-1.3B",
    superseded_by="Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

# ---------------------------------------------------------------------------
# SGPT-BE 125M — GPT-Neo 125M base, symmetric NLI
# ---------------------------------------------------------------------------

muennighoff__sgpt_125m_weightedmean_nli_bitfit = ModelMeta(
    name="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="23691c8b63a42b9a9796965c152feea82e25133c",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by=None,
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_weightedmean_nli = ModelMeta(
    name="Muennighoff/SGPT-125M-weightedmean-nli",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="9e7693f0f115b3ddecde41b574574e0750b3841c",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-nli",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_mean_nli_bitfit = ModelMeta(
    name="Muennighoff/SGPT-125M-mean-nli-bitfit",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="7f4eec6d68aae9464b71120cd4e29e98d6f6590a",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-mean-nli-bitfit",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_mean_nli = ModelMeta(
    name="Muennighoff/SGPT-125M-mean-nli",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="e3eae5208183fab1cd297be8f369b98654c77c02",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-mean-nli",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_lasttoken_nli = ModelMeta(
    name="Muennighoff/SGPT-125M-lasttoken-nli",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="5f48a2059f3684f5deaa752dca56694d63a154e7",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-lasttoken-nli",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_learntmean_nli = ModelMeta(
    name="Muennighoff/SGPT-125M-learntmean-nli",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="0385b7f25dd0729c60f8d7ea1b9290b529d63a40",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-learntmean-nli",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_mean_nli_linear5 = ModelMeta(
    name="Muennighoff/SGPT-125M-mean-nli-linear5",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="2d6e8ac05d13e9c5737c284781f8388659a28b85",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-mean-nli-linear5",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)

muennighoff__sgpt_125m_mean_nli_linearthenpool5 = ModelMeta(
    name="Muennighoff/SGPT-125M-mean-nli-linearthenpool5",
    model_type=["dense"],
    loader=sentence_transformers_loader,
    revision="dd054e39070dad240e5400b81eaf8f6d495bf564",
    release_date="2022-03-02",
    languages=["eng-Latn"],
    n_parameters=125_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=75,
    embed_dim=768,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/Muennighoff/sgpt",
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch"],
    reference="https://huggingface.co/Muennighoff/SGPT-125M-mean-nli-linearthenpool5",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=_NLI_DATASETS,
    adapted_from="EleutherAI/gpt-neo-125M",
    superseded_by="Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    citation=SGPT_CITATION,
)
