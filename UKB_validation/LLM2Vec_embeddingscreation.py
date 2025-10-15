import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any, Tuple, Optional

class EmbeddingProcessor:
    """
    A class to handle the processing of embeddings using various models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding processor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters.
        """
        self.config = config
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_embeddings(self, prepared_records: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """
        Process embeddings based on the specified model and settings.
        
        Args:
            prepared_records: DataFrame containing the records to process.
            kwargs: Dictionary of keyword arguments controlling the processing.
            
        Returns:
            DataFrame containing the processed embeddings.
        """
        if kwargs["calculate_embeddings"]:
            embedding_df = self._generate_embeddings(prepared_records, kwargs)
            
            # Save embeddings if specified
            if kwargs.get("save_embeddings", False):
                self._save_embeddings(embedding_df, kwargs)
                
            # Return early if disease unspecific and only calculating embeddings
            if kwargs["calculate_embeddings"] and kwargs["diseaseunspecific"]:
                return embedding_df
        else:
            
            if not kwargs["infer_all"]:
                embedding_df = self._load_single_embeddings(self.config["embeddingfile"])
            else:
                embedding_df = self._load_multiple_embeddings(
                    self.config["embeddingfile_qwen"],
                    self.config["embeddingfile_qwen3"],
                    self.config["embeddingfile_llm2vec"],
                    self.config["embeddingfile_nvembed"],
                    self.config["embeddingfile_clmbr"]
                )
                    
        return embedding_df
    
    def _generate_embeddings(self, prepared_records: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate embeddings based on the specified model.
        
        Args:
            prepared_records: DataFrame containing the records to process.
            kwargs: Dictionary of keyword arguments controlling the processing.
            
        Returns:
            DataFrame containing the generated embeddings.
        """
        model_name = kwargs["model"]
        
        if model_name == "LLM2Vec":
            return self._process_llm2vec(prepared_records, kwargs)
        elif model_name == "NVEmbed":
            return self._process_nvembed(prepared_records, kwargs)
        elif model_name == "Qwen":
            return self._process_qwen(prepared_records, kwargs)
        elif model_name == "Qwen3":
            return self._process_qwen(prepared_records, kwargs)
        elif model_name == "Llama":
            return self._process_llama(prepared_records, kwargs)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    
    def _process_llm2vec(self, prepared_records: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Process using LLM2Vec model."""
        from llm2vec import LLM2Vec
        
        model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="auto", #"cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=kwargs["tokenlength"],
            doc_max_length=kwargs["tokenlength"]
        )

        queries = prepared_records.queries.tolist()
        q_reps = model.encode(queries, batch_size=self.config["batch_size"], device=self.config["device"])

        print(q_reps.shape)
        embedding_df = pd.DataFrame()
        embedding_df["eid"] = prepared_records["eid"].astype(int)
        embedding_df["q_reps"] = list(q_reps)
        
        return embedding_df
    
    def _process_nvembed(self, prepared_records: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Process using NVEmbed model."""
        model = AutoModel.from_pretrained(
            "nvidia/NV-Embed-v2", 
            device_map="cuda" if torch.cuda.is_available() else "cpu", 
            trust_remote_code=True, 
            torch_dtype=torch.float16
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        task_name_to_instruct = self.config["instruction"]

        query_prefix = f"Instruct: {task_name_to_instruct}\nQuery: "
        queries = [row[1] for row in prepared_records["queries"]]
        max_length = kwargs["tokenlength"]

        with torch.no_grad():
            q_reps = model._do_encode(
                queries, 
                batch_size=self.config["batch_size"], 
                instruction=query_prefix, 
                max_length=max_length, 
                num_workers=1, 
                return_numpy=True
            )

        print(q_reps.shape)
        embedding_df = pd.DataFrame()
        embedding_df["eid"] = prepared_records["eid"].astype(int)
        embedding_df["q_reps"] = list(q_reps)
        
        return embedding_df
    
    def _process_qwen(self, prepared_records: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Process using Qwen model."""
        # Helper function for token pooling
        def last_token_pool(last_hidden_states, attention_mask):
            batch_size = last_hidden_states.shape[0]
            
            left_padding = (attention_mask[:, -1].sum() == batch_size)
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        # Prepare data
        task_name_to_instruct = self.config["instruction"]
        queries = [f"Instruct: {task_name_to_instruct}\nQuery: {query}" for query in [row[1] for row in prepared_records["queries"]]]

        # Load model components
        if(kwargs["model"] == "Qwen3"):
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
            model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B')
        else:
            tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
            #model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
            model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, torch_dtype=torch.float16)

        # Set up GPU with mixed precision
        model.to(self.device)

        # Enable mixed precision for faster computation on GPU
        amp_enabled = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

        # Process in batches efficiently
        max_length = kwargs["tokenlength"]
        batch_size = self.config["batch_size"]
        all_embeddings = []
        all_eids = prepared_records["eid"].astype(int)

        # Process in batches with optimized memory usage
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Processing Batches"):
                batch_queries = queries[i:i + batch_size]
                
                # Tokenize and move to GPU in one step
                batch_dict = tokenizer(
                    batch_queries, 
                    max_length=max_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.device)
                
                # Use mixed precision for faster computation
                if amp_enabled:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch_dict)
                        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                else:
                    outputs = model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Store embeddings efficiently
                all_embeddings.append(embeddings.cpu().numpy())

                # Free up GPU memory for the next batch
                del batch_dict, outputs, embeddings
                torch.cuda.empty_cache()

        # Concatenate results more efficiently
        final_embeddings = np.vstack(all_embeddings)

        # Create final dataframe
        embedding_df = pd.DataFrame({
            "eid": all_eids,
            "q_reps": list(final_embeddings)
        })
        
        return embedding_df
    
    def _process_llama(self, prepared_records: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Process using Llama model."""
        def initialize_model_and_tokenizer():
            model_id = "meta-llama/Llama-3.1-8B"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.eos_token_id

            return model, tokenizer

        def get_next_token_probabilities_batch(texts, target_words, model, tokenizer):
            results = []
            
            # Process texts in batches to avoid memory issues
            batch_size = 8  # Adjust based on your GPU memory
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=kwargs["tokenlength"]
                ).to(model.device)
                
                # Get model's output for batch
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                # Get probabilities for the last position of each sequence
                next_token_logits = logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Get token IDs for target words (do this once)
                target_token_ids = [
                    tokenizer.encode(word, add_special_tokens=False)[0] 
                    for word in target_words
                ]
                
                # Process each item in batch
                for probs in next_token_probs:
                    batch_results = {}
                    for word, token_id in zip(target_words, target_token_ids):
                        prob = probs[token_id].item()
                        batch_results[word] = prob
                    results.append(batch_results)
            
            return results

        # Initialize model and tokenizer once
        model, tokenizer = initialize_model_and_tokenizer()
        
        # Prepare all texts
        text_end = f"""### End of Electronic Healthcare Record \n
        If the risk factors and symptoms strongly indicate {self.config["disease"]} risk (>70% confidence), answer with Yes. If the risk factors and symptoms suggest low risk (<70% confidence), answer with No. Answer:"""
        
        texts = [text[0] + text[1] + text_end for text in prepared_records["queries"]]
        
        # Define target words
        target_words = ["yes", "no", "Yes", "No", "YES", "NO"]
        
        # Get probabilities for all texts
        all_probabilities = get_next_token_probabilities_batch(texts, target_words, model, tokenizer)
        
        # Calculate ratios
        ratio_list = []
        yes_list = []
        no_list = []
        for probabilities in tqdm(all_probabilities):
            p_yes = probabilities["Yes"] + probabilities["yes"] + probabilities["YES"]
            p_no = probabilities["No"] + probabilities["no"] + probabilities["NO"]
            ratio_list.append(p_yes / (p_yes + p_no))
            yes_list.append(p_yes)
            no_list.append(p_no)
        
        prepared_records["probability"] = ratio_list
        prepared_records["yes_proba"] = yes_list
        prepared_records["no_proba"] = no_list
        
        # Create a dataframe with the results
        embedding_df = pd.DataFrame({
            "eid": prepared_records["eid"].astype(int),
            "probability": ratio_list,
            "yes_proba": yes_list,
            "no_proba": no_list
        })
        
        return embedding_df
    
    def _save_embeddings(self, embedding_df: pd.DataFrame, kwargs: Dict[str, Any]) -> None:
        """
        Save embeddings to a file.
        
        Args:
            embedding_df: DataFrame containing the embeddings.
            kwargs: Dictionary of keyword arguments including the file path.
        """
        embedding_df_out = embedding_df.copy()
        embedding_df_out["q_reps"] = embedding_df_out["q_reps"].apply(lambda x: x.tolist())
        embedding_df_out.to_feather(self.config["embeddingfile"])
    
    def _load_single_embeddings(self, embedding_file: str) -> pd.DataFrame:
        """
        Load embeddings from a single file.
        
        Args:
            embedding_file: Path to the embedding file.
            
        Returns:
            DataFrame containing the loaded embeddings.
        """
        embedding_df = pd.read_feather(embedding_file)
        embedding_df["q_reps"] = embedding_df["q_reps"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        return embedding_df

    def _load_multiple_embeddings(self, qwen_file: str, qwen_file3: str, llm2vec_file: str, nvembed_file: str, clmbr_file: str) -> pd.DataFrame:
        """
        Load embeddings from multiple files and merge them.
        
        Args:
            qwen_file: Path to the Qwen embeddings file.
            llm2vec_file: Path to the LLM2Vec embeddings file.
            nvembed_file: Path to the NVEmbed embeddings file.
            clmbr_file: Path to the CLMBR embeddings file.
            
        Returns:
            DataFrame containing the merged embeddings.
        """
        # Load from individual files
        embedding_df_qwen = pd.read_feather(qwen_file)
        embedding_df_qwen3 = pd.read_feather(qwen_file3)
        embedding_df_llm2vec = pd.read_feather(llm2vec_file)
        if(self.config["useNVEmbed"]):
            embedding_df_nvembed = pd.read_feather(nvembed_file)
        embedding_df_clmbr = pd.read_feather(clmbr_file)

        # Convert to tensors and rename columns
        embedding_df_qwen["q_reps_qwen"] = embedding_df_qwen["q_reps"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        embedding_df_qwen.drop(columns=["q_reps"], inplace=True)

        embedding_df_qwen3["q_reps_qwen3"] = embedding_df_qwen3["q_reps"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        embedding_df_qwen3.drop(columns=["q_reps"], inplace=True)
        
        embedding_df_llm2vec["q_reps_llm2vec"] = embedding_df_llm2vec["q_reps"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        embedding_df_llm2vec.drop(columns=["q_reps"], inplace=True)
        
        if(self.config["useNVEmbed"]):
            embedding_df_nvembed["q_reps_nvembed"] = embedding_df_nvembed["q_reps"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
            embedding_df_nvembed.drop(columns=["q_reps"], inplace=True)
        
        embedding_df_clmbr["q_reps_clmbr"] = embedding_df_clmbr["q_reps"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        embedding_df_clmbr.drop(columns=["q_reps"], inplace=True)

        # Merge dataframes
        embedding_df = pd.merge(embedding_df_qwen, embedding_df_llm2vec, on='eid', how='inner')
        embedding_df = pd.merge(embedding_df, embedding_df_qwen3, on='eid', how='inner')
        if(self.config["useNVEmbed"]):
            embedding_df = pd.merge(embedding_df, embedding_df_nvembed, on='eid', how='inner')
        embedding_df = pd.merge(embedding_df, embedding_df_clmbr, on='eid', how='inner')
        
        # Select only the necessary columns
        columns = ["eid", "q_reps_qwen", "q_reps_qwen3", "q_reps_llm2vec", "q_reps_clmbr"]
        if(self.config["useNVEmbed"]):
            columns.append("q_reps_nvembed")
        embedding_df = embedding_df[columns]
        
        return embedding_df


# For convenience, create a function to get embeddings
def process_embeddings(prepared_records, config, **kwargs):
    """
    Process embeddings based on the specified model and settings.
    
    Args:
        prepared_records: DataFrame containing the records to process.
        config: Dictionary containing configuration parameters.
        **kwargs: Additional keyword arguments controlling the processing.
        
    Returns:
        DataFrame containing the processed embeddings.
    """
    processor = EmbeddingProcessor(config)
    return processor.process_embeddings(prepared_records, kwargs)