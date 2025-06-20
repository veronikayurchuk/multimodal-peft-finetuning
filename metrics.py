# Custom JSON-specific metrics for evaluating AI model predictions
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional, Union
from collections import Counter
import re
import difflib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk import edit_distance
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nltk.download('punkt_tab')

class JSONMetrics:
    """
    A class that implements various metrics for evaluating JSON predictions against ground truth.
    These metrics are particularly useful for evaluating AI models that generate structured data.
    """
    
    def __init__(self):
        """Initialize the JSONMetrics class."""
        pass
        
    def _flatten_json(self, json_obj, prefix=''):
        """
        Flatten a nested JSON object into a flat dictionary with dot notation.
        
        Args:
            json_obj: JSON object to flatten
            prefix: Prefix for keys in the flattened dictionary
            
        Returns:
            dict: Flattened dictionary
        """
        items = {}
        for k, v in json_obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.update(self._flatten_json(v, key))
                else:
                    items[key] = v
        return items
    
    def _get_keys_recursive(self, json_obj, prefix='', full_path=False):
        """
        Get all keys recursively from a JSON object.
        
        Args:
            json_obj: JSON object to get keys from
            prefix: Prefix for keys
            full_path: Whether to return full paths or just key names
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        keys = set()
        if isinstance(json_obj, dict):
            for k, v in json_obj.items():
                current_key = f"{prefix}.{k}" if prefix else k
                keys.add(current_key)
                if isinstance(v, (dict, list)):
                        keys.update(self._get_keys_recursive(v, current_key))
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                current_key = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    keys.update(self._get_keys_recursive(item, current_key))
        return keys
    
    
    def structure_similarity(self, pred: Dict, gt: Dict) -> float:
        """
        Calculate structure similarity by comparing keys in the JSON objects.
        This metric focuses on the structure rather than values.
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        gt_keys = self._get_keys_recursive(gt, full_path=True)
        pred_keys = self._get_keys_recursive(pred, full_path=True)
        
        if not gt_keys and not pred_keys:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = len(gt_keys.intersection(pred_keys))
        union = len(gt_keys.union(pred_keys))
        
        return intersection / union if union > 0 else 0.0
    
    def value_accuracy(self, pred: Dict, gt: Dict) -> float:
        """
        Calculate value accuracy by comparing values for matching keys.
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        gt_flat = self._flatten_json(gt)
        pred_flat = self._flatten_json(pred)
        
        # Find common keys
        common_keys = set(gt_flat.keys()).intersection(set(pred_flat.keys()))
        
        if not common_keys:
            return 0.0
        
        # Count exact matches
        matches = sum(1 for k in common_keys if str(gt_flat[k]) == str(pred_flat[k]))
        
        return matches / len(common_keys)
    
    def field_presence(self, pred: Dict, gt: Dict) -> float:
        """
        Calculate field presence score (recall of fields).
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        gt_keys = self._get_keys_recursive(gt)
        pred_keys = self._get_keys_recursive(pred)
        
        if not gt_keys:
            return 1.0
        
        # Calculate recall
        intersection = len(gt_keys.intersection(pred_keys))
        return intersection / len(gt_keys)
    
    def field_precision(self, pred: Dict, gt: Dict) -> float:
        """
        Calculate field precision score.
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        gt_keys = self._get_keys_recursive(gt)
        pred_keys = self._get_keys_recursive(pred)
        
        if not pred_keys:
            return 0.0
        
        # Calculate precision
        intersection = len(gt_keys.intersection(pred_keys))
        return intersection / len(pred_keys)
    
    def field_f1(self, pred: Dict, gt: Dict) -> float:
        """
        Calculate field F1 score (harmonic mean of precision and recall).
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        precision = self.field_precision(pred, gt)
        recall = self.field_presence(pred, gt)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using difflib's SequenceMatcher.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        return difflib.SequenceMatcher(None, str(str1), str(str2)).ratio()
    
    def numeric_value_error(self, pred_value: Union[str, int, float], gt_value: Union[str, int, float]) -> float:
        """
        Calculate normalized error for numeric values.
        Handles numeric values that might be represented as strings with commas, currency symbols, etc.
        
        Args:
            pred_value: Predicted value
            gt_value: Ground truth value
            
        Returns:
            float: Error score (lower is better)
        """
        def extract_number(value):
            if isinstance(value, (int, float)):
                return float(value)
            
            if isinstance(value, str):
                # Remove common non-numeric characters
                cleaned = re.sub(r'[^0-9.-]', '', value)
                try:
                    return float(cleaned)
                except (ValueError, TypeError):
                    return None
            
            return None
        
        pred_num = extract_number(pred_value)
        gt_num = extract_number(gt_value)
        
        if pred_num is None or gt_num is None:
            return 1.0  # Maximum error if not numeric
        
        if gt_num == 0:
            return 0.0 if pred_num == 0 else 1.0
        
        # Calculate normalized absolute error, capped at 1.0
        error = min(1.0, abs(pred_num - gt_num) / abs(gt_num))
        return error
    
    def semantic_similarity(self, pred_value: str, gt_value: str) -> float:
        """
        Calculate semantic similarity for text values using BLEU score.
        
        Args:
            pred_value: Predicted text value
            gt_value: Ground truth text value
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if not isinstance(pred_value, str) or not isinstance(gt_value, str):
            return 0.0 if pred_value != gt_value else 1.0
            
        if pred_value == gt_value:
            return 1.0
            
        # Tokenize the strings
        reference = [nltk.word_tokenize(gt_value.lower())]
        hypothesis = nltk.word_tokenize(pred_value.lower())
        
        # If either is empty after tokenization, use string similarity
        if not reference[0] or not hypothesis:
            return self.string_similarity(pred_value, gt_value)
        
        # Calculate BLEU score with smoothing
        try:
            smoothie = SmoothingFunction().method1
            return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
        except:
            # Fallback to string similarity if BLEU fails
            return self.string_similarity(pred_value, gt_value)
    
    def value_similarity(self, pred: Dict, gt: Dict) -> float:
        """
        Calculate overall value similarity by comparing values for matching keys,
        using appropriate similarity metrics based on value types.
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        gt_flat = self._flatten_json(gt)
        pred_flat = self._flatten_json(pred)
        
        # Find common keys
        common_keys = set(gt_flat.keys()).intersection(set(pred_flat.keys()))
        
        if not common_keys:
            return 0.0
        
        similarities = []
        
        for k in common_keys:
            gt_val = gt_flat[k]
            pred_val = pred_flat[k]
            
            # Skip None values
            if gt_val is None and pred_val is None:
                similarities.append(1.0)
                continue
                
            if gt_val is None or pred_val is None:
                similarities.append(0.0)
                continue
            
            # Try to extract numbers for numeric comparison
            gt_num = re.sub(r'[^0-9.-]', '', str(gt_val)) if isinstance(gt_val, str) else gt_val
            pred_num = re.sub(r'[^0-9.-]', '', str(pred_val)) if isinstance(pred_val, str) else pred_val
            
            try:
                gt_num = float(gt_num)
                pred_num = float(pred_num)
                # Use numeric error for numeric values
                error = self.numeric_value_error(pred_val, gt_val)
                similarities.append(1.0 - error)
            except (ValueError, TypeError):
                # Use semantic similarity for text values
                similarities.append(self.semantic_similarity(str(pred_val), str(gt_val)))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def edit_distance(self, pred: Dict, gt: Dict) -> float:
        """
        Calculates Levenhstein edit distance. 
        This quantifies how much we would need to edit the predicted token sequence to get the target sequence. 
        The lower - the better.
        Its optimal value is 0 (which means, no edits need to be made).
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            Dict[str, float]: Dictionary containing individual metrics and an overall score
        """
        edit_distance_metric = edit_distance(str(pred), str(gt)) / max(len(str(pred)), len(str(gt)))
        return edit_distance_metric
    
    def calculate_overall_score(self, pred: Dict, gt: Dict) -> Dict[str, float]:
        """
        Calculate an overall evaluation score combining multiple metrics.
        
        Args:
            pred: Predicted JSON
            gt: Ground truth JSON
            
        Returns:
            Dict[str, float]: Dictionary containing individual metrics and an overall score
        """
        structure = self.structure_similarity(pred, gt)
        value_acc = self.value_accuracy(pred, gt)
        field_rec = self.field_presence(pred, gt)
        field_prec = self.field_precision(pred, gt)
        f1 = self.field_f1(pred, gt)
        value_sim = self.value_similarity(pred, gt)
        edit_distance_metric = self.edit_distance(pred, gt)
        
        # Calculate weighted overall score
        # You can adjust these weights based on what's most important for your use case
        overall = (
            0.20 * structure +  # Structure is very important
            0.20 * value_acc +  # Exact value matches
            0.15 * f1 +  # Field F1 score
            0.25 * value_sim + # Value similarity (most flexible metric)
            0.20 * edit_distance_metric  # Edit distance (flexible metric)
        )
        
        return {
            'structure_similarity': structure,
            'value_accuracy': value_acc,
            'field_recall': field_rec,
            'field_precision': field_prec,
            'field_f1': f1,
            'value_similarity': value_sim,
            'edit_distance': edit_distance_metric,
            'overall_score': overall
        }


if __name__ == '__main__':
    gt_json = {'menu': [{'nm': 'PEARL CHOCO TEA', 'cnt': '1', 'price': '17.000'}, {'nm': 'GREEN TEA LYCHEE', 'cnt': '1', 'price': '18.000'}, {'nm': 'TUTUP SEAL', 'cnt': '2', 'price': '0'}, {'nm': 'CUP 14 OZ', 'cnt': '2', 'price': '0'}], 'sub_total': {'subtotal_price': '35.000'}, 'total': {'total_price': '35.000', 'cashprice': '100.000', 'changeprice': '65.000', 'menuqty_cnt': '6'}}
    preds_json = {'menu': [{'nm': 'PEARL CHOCO TEA', 'cnt': '1', 'price': '17.000'}, {'nm': 'GREEN TEA LYCHEE', 'cnt': '1', 'price': '18.000'}, {'nm': 'TOFU SEA', 'cnt': '2', 'price': '0'}, {'nm': 'CAP KOK', 'cnt': '2', 'price': '0'}], 'sub_total': {'subtotal_price': '35.000'}, 'total': {'total_price': '35.000', 'cashprice': '100.000', 'changeprice': '65.000'}}

    metrics = JSONMetrics()

    test_metric = metrics.calculate_overall_score(preds_json, gt_json)
    print(test_metric)
