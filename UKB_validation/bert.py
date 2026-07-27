from abc import ABC, abstractmethod
from math import e
from typing import List, Any, Optional
from numpy.typing import NDArray
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from tqdm import tqdm
from typing import Tuple
import hashlib
import os
import pickle
# NOTE: workaround for LLM2Vec models that are not compatible with most recent transformers library for ModernBERT, Qwen3
from llm2vec import LLM2Vec
import torch



class TextsDataset(TorchDataset):
    def __init__(self, texts):
        self.texts = texts 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
    
class LLMEncoder(ABC):
    def __init__(self, embedding_size: int, model_max_input_length: int, max_input_length: int) -> None:
        self.embedding_size = embedding_size
        self.max_input_length = min(model_max_input_length, max_input_length)

        # Use simple heuristic to determine batch size
        # TODO: Adapt based on available GPU memory
        def determine_llm_batch_size():
            # For max_input_length = 8192 (2 for 40 GB, 16 (llama), 8 for Qwen for 80 GB)
            batch_size = 8

            if self.__class__.__name__.startswith('LLM2VecLlama3'):
                batch_size = 16
        
            if max_input_length > 32768:
                batch_size = 1
            elif max_input_length > 8192:
                batch_size = 2
            elif max_input_length == 512:
                batch_size = 64
        
            # Qwen3 needs smaller batch size for 4b and 8b models for 8192 input length
            if self.__class__.__name__.startswith('Qwen3Embedding_4B') and max_input_length == 8192:
                batch_size = 4
            elif self.__class__.__name__.startswith('Qwen3Embedding_8B') and max_input_length == 8192:
                batch_size = 2
                
            # BERT models can use larger batch size, since they are generally smaller and use up to 512 tokens
            if self.__class__.__name__ == 'BertEncoder':
                    batch_size = 512
                
            return batch_size
        
        
        self.batch_size: int = determine_llm_batch_size()
        
        # Ensure that tokenizer and model are set, but this is done in subclasses
        self.tokenizer = None
        self.model = None   

    def add_instruction(self, instruction: str, text: str) -> Any:
        # Per default: ignore instruction
        return text
    
    def get_chunked_dataset(self, texts: List[str], tokenizer, max_chunks=None) -> Tuple[List[str], List[int]]:
        # Create chunks of size max_input_length tokens for each text
        batch_size=8192 
        max_input_length = self.max_input_length - 8  # Subtract 8 to account for potential special tokens
        
        all_chunks = []
        chunk_counts = []
        
        start_idx = 0
        while start_idx < len(texts):
            print(f"  Chunking batch {start_idx // batch_size + 1} of {len(texts) // batch_size + 1}")
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_offsets = tokenizer(batch_texts, add_special_tokens=False, return_offsets_mapping=True, truncation=False, padding=False)["offset_mapping"]
        
            for text, offsets in zip(batch_texts, batch_offsets):
                num_offsets = len(offsets)
                
                # Pre-limit how many indices we'll iterate over, so we generate at most `max_chunks` slices.
                if max_chunks is not None:
                    limit = max_chunks * max_input_length
                    end = min(num_offsets, limit)
                else:
                    end = num_offsets
                
                text_chunks = [
                    text[offsets[i][0]:offsets[min(i + max_input_length, num_offsets) - 1][1]]
                    for i in range(0, end, max_input_length)
                ]
                    
                chunk_counts.append(len(text_chunks))
                all_chunks.extend(text_chunks)
                
            start_idx = end_idx
        print()
            
        return all_chunks, chunk_counts
    
    def get_averaged_chunks(self, all_embeddings: NDArray[Any], chunk_counts: List[int]) -> NDArray[Any]:
        current_index = 0
        averaged_embeddings = []
        for count in chunk_counts:
            # Handle case of empty chunk, which can happen if text is empty 
            if count == 0:
                averaged_embeddings.append(np.zeros(self.embedding_size))
                continue
            chunk_embeddings = all_embeddings[current_index:current_index + count]
            averaged_embeddings.append(np.mean(chunk_embeddings, axis=0))
            current_index += count
        return np.array(averaged_embeddings)
    
    def get_concatenated_chunks(self, all_embeddings: NDArray[Any], chunk_counts: List[int], max_chunks: int, per_chunk_embedding_size: int) -> NDArray[Any]:
        current_index = 0
        concatenated_embeddings = []
        for count in chunk_counts:
            # Handle case of empty chunk, which can happen if text is empty 
            if count == 0:
                concatenated_embeddings.append(np.zeros(max_chunks * per_chunk_embedding_size))
                continue
            chunk_embeddings = all_embeddings[current_index:current_index + count]
            current_index += count

            if count < max_chunks:
                pad = np.zeros((max_chunks - count, per_chunk_embedding_size))
                chunk_embeddings = np.concatenate([chunk_embeddings, pad], axis=0)
            else:
                chunk_embeddings = chunk_embeddings[:max_chunks]

            concatenated_embeddings.append(chunk_embeddings.reshape(max_chunks * per_chunk_embedding_size))
        return np.array(concatenated_embeddings)
        
    @abstractmethod
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        pass
            

class BERTLLMEncoder(LLMEncoder):
    def __init__(self, embedding_size: int, model_max_input_length: int, max_input_length: int) -> None:
        super().__init__(embedding_size, model_max_input_length, max_input_length)


class BertEncoder(BERTLLMEncoder):
    
    def __init__(self, max_input_length: int, bert_identifier: str, embedding_size: int, model_max_input_length: int, concat_embeddings: bool = False, **kwargs) -> None:
        # use variable bert_identifier, embedding_size, model_max_input_length to allow for different BERT models
        super().__init__(embedding_size=embedding_size, model_max_input_length=model_max_input_length, max_input_length=max_input_length)  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(bert_identifier)

        self.concat_embeddings = concat_embeddings
        self.per_chunk_embedding_size = embedding_size
        if self.concat_embeddings:
            BASE_INPUT_LENGTH = 8192
            max_chunks = BASE_INPUT_LENGTH // self.max_input_length
            self.embedding_size = self.per_chunk_embedding_size * max_chunks

        # Prefer safetensors, but allow fallback to PyTorch bin weights
        try:
            self.model = AutoModel.from_pretrained(bert_identifier, use_safetensors=True).to(self.device)
        except Exception as e_safetensors:
            print(
                f"[WARN] Could not load safetensors for '{bert_identifier}'. "
                f"Falling back to PyTorch weights (.bin). Error was: {e_safetensors}"
            )
            self.model = AutoModel.from_pretrained(bert_identifier, use_safetensors=False,).to(self.device)

        # Enable multi-gpu support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)

    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        # Use multiples of this base input length to determine the max number of chunks, e.g. for 2k chunks use max number of 4
        BASE_INPUT_LENGTH = 8192
        max_chunks = BASE_INPUT_LENGTH // self.max_input_length
        # To save memory, shorten texts to BASE_INPUT_LENGTH * 8 characters as a very loose upper bound for the number of tokens
        inputs = [text[:BASE_INPUT_LENGTH * 8] for text in inputs]
        
        # Create chunks of the inputs before calling the superclass encode method
        num_inputs = len(inputs)
        print(f"Creating chunks for {num_inputs} inputs of size {self.max_input_length} (max_chunks: {max_chunks}).")
        inputs, chunk_counts = self.get_chunked_dataset(inputs, self.tokenizer, max_chunks=max_chunks)
        
        # For small models increase batch size
        base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        hidden_size = base_model.config.hidden_size
        if hidden_size == 768:
            self.batch_size = self.batch_size * 2
            
        print(f"Encoding {len(inputs)} chunks with batch size {self.batch_size}.")
        dataloader = DataLoader(TextsDataset(inputs), batch_size=self.batch_size, shuffle=False, collate_fn=lambda batch: batch)
        
        all_embeddings_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding Chunks"):
                inputs_dict = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_input_length, return_tensors='pt')
                inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
                outputs = self.model(**inputs_dict, output_hidden_states=True)
                # Get average of all hidden states in the last hidden layer
                # Shown to be superior to cls token or max (https://arxiv.org/pdf/1908.10084)
                # For implementation see: https://github.com/autoliuweijie/BERT-whitening-pytorch/blob/b5cfbd606bd19fc3b3adf9e074dc0bfd830ef597/all_utils.py#L33
                # Want to reproduce Jiang et al. Health system-scale language models are all-purpose prediction engines 2023. However, unclear what MLM classification head exactly means.
                last_avg_embedding = outputs.hidden_states[-1].mean(dim=1)
                all_embeddings_list.append(last_avg_embedding.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings_list, axis=0)

        if self.concat_embeddings:
            all_embeddings = self.get_concatenated_chunks(all_embeddings, chunk_counts, max_chunks, self.per_chunk_embedding_size)
        else:
            all_embeddings = self.get_averaged_chunks(all_embeddings, chunk_counts)
        assert len(all_embeddings) == num_inputs

        return all_embeddings


class TextEncoder:
    def __init__(self, encoder: LLMEncoder):
        self.encoder = encoder
        
    def _store_or_check_fingerprint(self, inputs: List, cache_dir: str) -> None:
        fingerprint_file = os.path.join(cache_dir, "cache_fingerprint.txt")
        
        # Generate fingerprint
        hasher = hashlib.sha256()
        for input in inputs:
            hasher.update((str(input)).encode('utf-8'))
        fingerprint = str(len(inputs)) + '-' + hasher.hexdigest()
        
        # Check for existing fingerprint
        if os.path.exists(fingerprint_file):
            with open(fingerprint_file, "r") as f:
                existing_fingerprint = f.read().strip()
            if existing_fingerprint != fingerprint:
                raise ValueError("Cache fingerprint does not match. Data inconsistency detected.")
        else:
            with open(fingerprint_file, "w") as f:
                f.write(fingerprint)

    def _get_cache_files(self, cache_dir: str) -> List[str]:
        file_names = os.listdir(cache_dir)
        return [f for f in file_names if f.startswith('cache_') and f.endswith('.pkl')]
    
    def _delete_all_cache_files(self, cache_dir: str) -> None:
        cache_files = self._get_cache_files(cache_dir)
        for cache_file in cache_files:
            os.remove(os.path.join(cache_dir, cache_file))
    
    def encode_texts(self, instructions: List[str], texts: List[str], cache_dir: Optional[str] = None) -> NDArray[Any]:
        # Add instructions to texts
        if all([instruction is None or len(instruction) == 0 for instruction in instructions]):
            inputs = texts
        else:
            inputs = [self.encoder.add_instruction(instruction, text) for instruction, text in zip(instructions, texts)]
        
        return self.encoder._encode(inputs)
        