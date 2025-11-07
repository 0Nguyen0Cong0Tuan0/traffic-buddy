"""
Evaluation metrics for Vietnamese Traffic VQA
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculate various metrics for VQA evaluation.
    """

    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.question_types = []
        self.sample_ids = []
    
    def add_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        question_types: List[str] = None,
        sample_ids: List[str] = None
    ):
        """
        Add a batch of predictions
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)

        if question_types:
            self.question_types.extend(question_types)
        if sample_ids:
            self.sample_ids.extend(sample_ids)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        """
        if not self.predictions:
            return {}

        # Overall accuracy
        accuracy = self._compute_accuracy(self.predictions, self.ground_truths)
        
        # Per-answer accuracy
        per_answer_acc = self._compute_per_answer_accuracy()

        # Per-question-type accuracy (if available)
        per_type_acc = {}
        if self.question_types:
            per_type_acc = self._compute_per_question_type_accuracy()
        
        # Confusion matrix
        confusion = self._compute_confusion_matrix()

        metrics = {
            'accuracy': accuracy,
            'per_answer_accuracy': per_answer_acc,
            'per_type_accuracy': per_type_acc,
            'confusion_matrix': confusion,
            'total_samples': len(self.predictions)
        }

        return metrics

    def _compute_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> float:
        """
        Compute overall accuracy
        """
        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        accuracy = correct / len(predictions) if predictions else 0.0
        return accuracy

    def _compute_per_answer_accuracy(self) -> Dict[str, float]:
        """
        Compute accuracy for each answer option (A, B, C, D)        
        """
        per_answer = defaultdict(lambda: {'correct': 0, 'total': 0})

        for pred, gt in zip(self.predictions, self.ground_truths):
            per_answer[gt]['total'] += 1
            if pred == gt:
                per_answer[gt]['correct'] += 1
        
        per_answer_acc = {}
        for answer, stats in per_answer.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            per_answer_acc[answer] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return per_answer_acc
    
    def _compute_per_type_accuracy(self) -> Dict[str, float]:
        """
        Compute accuracy for each question type
        """
        per_type = defaultdict(lambda: {'correct': 0, 'total': 0})

        for pred, gt, qtype in zip(self.predictions, self.ground_truths, self.question_types):
            per_type[qtype]['total'] += 1
            if pred == gt:
                per_type[qtype]['correct'] += 1
        
        per_type_acc = {}
        for qtype, stats in per_type.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            per_type_acc[qtype] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return per_type_acc
    
    def _compute_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Compute confusion matrix
        """
        labels = sorted(set(self.ground_truths + self.predictions))

        confusion = {label: {l: 0 for l in labels} for label in labels}

        for pred, gt in zip(self.predictions, self.ground_truths):
            confusion[gt][pred] += 1
        
        return confusion
    
    def print_report(self):
        """
        Print detailed evaluation report
        """
        metrics = self.compute()
        
        logger.info("\n" + "="*60)
        logger.info("Evaluation Report")
        logger.info("="*60)
        
        # Overall accuracy
        logger.info(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({int(metrics['accuracy'] * metrics['total_samples'])}/{metrics['total_samples']})")
        
        # Per-answer accuracy
        logger.info("\nPer-Answer Accuracy:")
        for answer in sorted(metrics['per_answer_accuracy'].keys()):
            stats = metrics['per_answer_accuracy'][answer]
            logger.info(f"  {answer}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        
        # Per-type accuracy
        if metrics['per_type_accuracy']:
            logger.info("\nPer-Question-Type Accuracy:")
            for qtype in sorted(
                metrics['per_type_accuracy'].keys(),
                key=lambda x: metrics['per_type_accuracy'][x]['accuracy'],
                reverse=True
            ):
                stats = metrics['per_type_accuracy'][qtype]
                logger.info(f"  {qtype}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
        
        # Confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info("(Rows: Ground Truth, Columns: Predictions)")
        
        confusion = metrics['confusion_matrix']
        labels = sorted(confusion.keys())
        
        # Header
        header = "GT\\Pred  " + "  ".join(f"{label:>6}" for label in labels)
        logger.info(header)
        
        # Rows
        for gt_label in labels:
            row = f"{gt_label:>7}  " + "  ".join(
                f"{confusion[gt_label][pred_label]:>6}" for pred_label in labels
            )
            logger.info(row)
        
        logger.info("="*60)
    
    def get_error_analysis(
        self,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Get samples with incorrect predictions for error analysis
        """
        errors = []

        for i, (pred, gt, sid) in enumerate(zip(
            self.predictions, self.ground_truths, self.sample_ids
        )):
            if pred != gt:
                errors.append({
                    'sample_id': sid,
                    'prediction': pred,
                    'ground_truth': gt,
                    'question_type': self.question_types[i] if self.question_types else None
                })
        
        return errors[:top_k]
    
    def reset(self):
        """
        Reset metrics calculator
        """
        self.predictions = []
        self.ground_truths = []
        self.sample_ids = []
        self.question_types = []

def extract_question_type(question: str) -> str:
    """
    Extract question type from question text
    """
    question_lower = question.lower()
    
    if 'biển' in question_lower or 'sign' in question_lower:
        return 'traffic_sign'
    elif 'rẽ' in question_lower or 'turn' in question_lower:
        return 'turn_direction'
    elif 'làn' in question_lower or 'lane' in question_lower:
        return 'lane'
    elif 'tốc độ' in question_lower or 'speed' in question_lower:
        return 'speed'
    else:
        return 'other'