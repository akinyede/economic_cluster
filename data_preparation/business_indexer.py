#!/usr/bin/env python3
"""Efficient business matching and indexing for 525K+ businesses"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import re
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    source_idx: int
    target_idx: int
    match_score: float
    match_type: str  # 'exact', 'fuzzy', 'phone', 'address'

class BusinessIndexer:
    """Fast multi-index matching for business datasets"""
    
    def __init__(self):
        self.indices = {
            'name': {},
            'phone': {},
            'address': {},
            'name_soundex': {},
            'name_tokens': defaultdict(set),
        }
        self.business_data = None
        
    def build_indices(self, df: pd.DataFrame) -> None:
        """Build multiple indices for fast business matching"""
        logger.info(f"Building indices for {len(df)} businesses...")
        
        self.business_data = df.copy()
        
        # Build name index
        for idx, row in df.iterrows():
            # Exact name
            name_key = self._normalize_name(row.get('name', ''))
            if name_key:
                if name_key not in self.indices['name']:
                    self.indices['name'][name_key] = []
                self.indices['name'][name_key].append(idx)
            
            # Name tokens (for partial matching)
            tokens = self._tokenize_name(row.get('name', ''))
            for token in tokens:
                self.indices['name_tokens'][token].add(idx)
            
            # Soundex
            soundex_key = self._soundex(name_key)
            if soundex_key:
                if soundex_key not in self.indices['name_soundex']:
                    self.indices['name_soundex'][soundex_key] = []
                self.indices['name_soundex'][soundex_key].append(idx)
            
            # Phone
            phone_key = self._normalize_phone(row.get('phone', ''))
            if phone_key:
                if phone_key not in self.indices['phone']:
                    self.indices['phone'][phone_key] = []
                self.indices['phone'][phone_key].append(idx)
            
            # Address
            addr_key = self._normalize_address(row.get('address', ''))
            if addr_key:
                if addr_key not in self.indices['address']:
                    self.indices['address'][addr_key] = []
                self.indices['address'][addr_key].append(idx)
        
        logger.info(f"  Built {len(self.indices['name'])} unique name keys")
        logger.info(f"  Built {len(self.indices['phone'])} unique phone keys")
        logger.info(f"  Built {len(self.indices['address'])} unique address keys")
        logger.info(f"  Built {len(self.indices['name_soundex'])} unique soundex keys")
        logger.info(f"  Built {len(self.indices['name_tokens'])} unique name tokens")
    
    def find_matches(self, target_df: pd.DataFrame, threshold: float = 0.8) -> List[MatchResult]:
        """Find matching businesses between datasets"""
        matches = []
        unmatched = []
        
        logger.info(f"Finding matches for {len(target_df)} businesses...")
        
        for target_idx, row in target_df.iterrows():
            best_match = self._find_best_match(row, threshold)
            
            if best_match:
                matches.append(best_match)
            else:
                unmatched.append(target_idx)
        
        logger.info(f"  Found {len(matches)} matches")
        logger.info(f"  {len(unmatched)} businesses unmatched")
        
        # Analyze match quality
        if matches:
            match_types = {}
            for m in matches:
                if m.match_type not in match_types:
                    match_types[m.match_type] = 0
                match_types[m.match_type] += 1
            
            logger.info("  Match types distribution:")
            for mtype, count in match_types.items():
                logger.info(f"    {mtype}: {count} ({count/len(matches)*100:.1f}%)")
            
            avg_score = np.mean([m.match_score for m in matches])
            logger.info(f"  Average match score: {avg_score:.3f}")
        
        return matches
    
    def _find_best_match(self, target_row: pd.Series, threshold: float) -> Optional[MatchResult]:
        """Find best matching business from indices"""
        candidates = set()
        
        # 1. Try exact name match
        name_key = self._normalize_name(target_row.get('name', ''))
        if name_key and name_key in self.indices['name']:
            for idx in self.indices['name'][name_key]:
                candidates.add((idx, 1.0, 'exact_name'))
                return MatchResult(idx, target_row.name, 1.0, 'exact_name')
        
        # 2. Try phone match
        phone_key = self._normalize_phone(target_row.get('phone', ''))
        if phone_key and phone_key in self.indices['phone']:
            for idx in self.indices['phone'][phone_key]:
                candidates.add((idx, 0.95, 'phone'))
                # Phone matches are very reliable
                return MatchResult(idx, target_row.name, 0.95, 'phone')
        
        # 3. Try address match
        addr_key = self._normalize_address(target_row.get('address', ''))
        if addr_key and addr_key in self.indices['address']:
            for idx in self.indices['address'][addr_key]:
                # Verify name similarity for address matches
                source_name = self._normalize_name(self.business_data.loc[idx, 'name'])
                similarity = self._string_similarity(name_key, source_name)
                if similarity > 0.7:
                    candidates.add((idx, 0.9 * similarity, 'address'))
        
        # 4. Try soundex match
        soundex_key = self._soundex(name_key)
        if soundex_key and soundex_key in self.indices['name_soundex']:
            for idx in self.indices['name_soundex'][soundex_key]:
                # Calculate actual string similarity
                source_name = self._normalize_name(self.business_data.loc[idx, 'name'])
                similarity = self._string_similarity(name_key, source_name)
                if similarity > threshold:
                    candidates.add((idx, similarity * 0.85, 'soundex'))
        
        # 5. Try token-based matching
        tokens = self._tokenize_name(target_row.get('name', ''))
        if len(tokens) >= 2:
            token_matches = set()
            for token in tokens:
                if token in self.indices['name_tokens']:
                    token_matches.update(self.indices['name_tokens'][token])
            
            # Score based on token overlap
            for idx in token_matches:
                source_tokens = self._tokenize_name(self.business_data.loc[idx, 'name'])
                if source_tokens:
                    overlap = len(tokens.intersection(source_tokens))
                    if overlap >= 2:
                        score = overlap / max(len(tokens), len(source_tokens))
                        if score > threshold * 0.8:
                            candidates.add((idx, score * 0.8, 'token'))
        
        # Select best candidate
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            if best[1] >= threshold:
                return MatchResult(best[0], target_row.name, best[1], best[2])
        
        return None
    
    def _normalize_name(self, name: str) -> str:
        """Normalize business name for matching"""
        if not name:
            return ''
        
        # Convert to lowercase
        normalized = str(name).lower()
        
        # Remove common suffixes
        suffixes = ['inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'co', 
                   'incorporated', 'limited', 'enterprises', 'associates']
        for suffix in suffixes:
            normalized = normalized.replace(f' {suffix}', '')
            normalized = normalized.replace(f' {suffix}.', '')
            normalized = normalized.replace(f' {suffix},', '')
        
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _tokenize_name(self, name: str) -> Set[str]:
        """Extract meaningful tokens from business name"""
        normalized = self._normalize_name(name)
        if not normalized:
            return set()
        
        # Split into tokens
        tokens = set(normalized.split())
        
        # Remove stop words
        stop_words = {'the', 'and', 'of', 'in', 'at', 'by', 'for', 'to', 'a', 'an'}
        tokens = tokens - stop_words
        
        # Only keep tokens with 3+ characters
        tokens = {t for t in tokens if len(t) >= 3}
        
        return tokens
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to digits only"""
        if not phone:
            return ''
        
        # Keep only digits
        digits = re.sub(r'\D', '', str(phone))
        
        # Handle 10-digit US numbers
        if len(digits) == 10:
            return digits
        elif len(digits) == 11 and digits[0] == '1':
            return digits[1:]  # Remove country code
        
        return ''
    
    def _normalize_address(self, address: str) -> str:
        """Normalize address for matching"""
        if not address:
            return ''
        
        normalized = str(address).lower()
        
        # Standardize common abbreviations
        replacements = {
            ' street': ' st',
            ' avenue': ' ave',
            ' road': ' rd',
            ' boulevard': ' blvd',
            ' drive': ' dr',
            ' lane': ' ln',
            ' court': ' ct',
            ' circle': ' cir',
            ' place': ' pl',
            ' north': ' n',
            ' south': ' s',
            ' east': ' e',
            ' west': ' w',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Remove apartment/suite numbers
        normalized = re.sub(r'(apt|apartment|suite|ste|unit|#)\s*\w+', '', normalized)
        
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _soundex(self, name: str) -> str:
        """Generate soundex code for phonetic matching"""
        if not name:
            return ''
        
        name = name.upper()
        soundex = name[0]
        
        # Soundex character mappings
        mappings = {
            'BFPV': '1',
            'CGJKQSXZ': '2',
            'DT': '3',
            'L': '4',
            'MN': '5',
            'R': '6'
        }
        
        for char in name[1:]:
            for key, value in mappings.items():
                if char in key:
                    if len(soundex) == 1 or soundex[-1] != value:
                        soundex += value
                    break
        
        # Pad with zeros and truncate to 4 characters
        soundex = (soundex + '000')[:4]
        
        return soundex
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein ratio"""
        if not s1 or not s2:
            return 0.0
        
        # Simple character-based similarity
        if s1 == s2:
            return 1.0
        
        # Calculate Levenshtein distance
        len1, len2 = len(s1), len(s2)
        dist = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dist[i][0] = i
        for j in range(len2 + 1):
            dist[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                dist[i][j] = min(
                    dist[i-1][j] + 1,      # deletion
                    dist[i][j-1] + 1,      # insertion
                    dist[i-1][j-1] + cost  # substitution
                )
        
        # Calculate similarity ratio
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0
        
        return 1.0 - (dist[len1][len2] / max_len)
    
    def merge_matched_data(self, source_df: pd.DataFrame, target_df: pd.DataFrame, 
                          matches: List[MatchResult], prefix: str = 'kc_') -> pd.DataFrame:
        """Merge matched data from target into source DataFrame"""
        logger.info(f"Merging {len(matches)} matched records...")
        
        # Create a copy of source DataFrame
        merged_df = source_df.copy()
        
        # Add new columns from target
        target_cols = [col for col in target_df.columns 
                      if col not in ['name', 'address', 'phone', 'city', 'state', 'zip']]
        
        for col in target_cols:
            merged_df[f'{prefix}{col}'] = None
        
        # Add match metadata columns
        merged_df[f'{prefix}match_score'] = 0.0
        merged_df[f'{prefix}match_type'] = ''
        
        # Merge matched data
        for match in matches:
            source_idx = match.source_idx
            target_idx = match.target_idx
            
            # Copy target data to source row
            for col in target_cols:
                merged_df.loc[source_idx, f'{prefix}{col}'] = target_df.loc[target_idx, col]
            
            # Add match metadata
            merged_df.loc[source_idx, f'{prefix}match_score'] = match.match_score
            merged_df.loc[source_idx, f'{prefix}match_type'] = match.match_type
        
        # Log merge statistics
        matched_count = (merged_df[f'{prefix}match_score'] > 0).sum()
        logger.info(f"  Successfully merged data for {matched_count} businesses")
        logger.info(f"  Match rate: {matched_count / len(source_df) * 100:.1f}%")
        
        return merged_df