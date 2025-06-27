#!/usr/bin/env python3
"""
Comprehensive Book Processing Pipeline for Trading Knowledge Extraction

This module processes trading and finance books to extract:
- Core concepts and methodologies
- Mathematical frameworks and algorithms
- Implementation strategies and code patterns
- Cross-book knowledge synthesis and connections
- Testable hypotheses and trading strategies
"""

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import requests

# Add project root to path
sys.path.append('/home/scott/TradeKnowledge')

from src.compression.llmlingua_compressor import CompressionResult, LLMLinguaCompressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConceptualFramework:
    """Represents a key conceptual framework extracted from books"""
    name: str
    description: str
    mathematical_foundation: str
    implementation_approach: str
    prerequisites: list[str]
    applications: list[str]
    book_source: str
    page_references: list[int]
    complexity_level: str  # beginner, intermediate, advanced
    testable_hypothesis: str | None = None


@dataclass
class TradingStrategy:
    """Represents a trading strategy extracted from books"""
    name: str
    description: str
    entry_criteria: str
    exit_criteria: str
    risk_management: str
    time_horizon: str
    asset_classes: list[str]
    data_requirements: list[str]
    expected_performance: str
    implementation_complexity: str
    book_source: str
    validation_method: str


@dataclass
class BookAnalysis:
    """Complete analysis results for a processed book"""
    book_title: str
    book_path: str
    processing_date: datetime
    total_pages: int
    conceptual_frameworks: list[ConceptualFramework]
    trading_strategies: list[TradingStrategy]
    key_insights: list[str]
    mathematical_content: list[str]
    implementation_examples: list[str]
    data_requirements: list[str]
    cross_references: list[str]
    knowledge_level: str  # foundational, intermediate, advanced
    practical_value_score: float  # 0-1 scale
    implementation_readiness: float  # 0-1 scale


class ComprehensiveBookProcessor:
    """Enhanced book processor for comprehensive trading knowledge extraction"""

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "qwen2.5-coder:7b"
        self.compressor = LLMLinguaCompressor(target_compression=0.8)  # Quality-focused compression

        # Quality control configuration
        self.max_retries = 3
        self.validation_enabled = True

        # Book processing phases
        self.processing_phases = {
            'foundational': [
                'Trading_Systems_and_Methods.pdf',
                'Yves Hilpisch - Python for Algorithmic Trading_ From Idea to Cloud Deployment-O\'Reilly Media (2020).pdf',
                'financial-modeling-under-non-gaussian-distributions.pdf'
            ],
            'ml_applications': [
                'Guillaume_Coqueret_Tony_Guida_-_Machine_Learning_for_Factor_Investing__Python_Version-CRC_Press_2023.pdf',
                'Detecting_regime_change_in_computational_finance_data_science_machine_learning_and_algorithmic_trading_by_Chen_Jun_Tsang_Edward_z-lib.org.pdf',
                'DiversifiedManagedFutures.pdf'
            ],
            'advanced_techniques': [
                'high-frequency-trading-a-practical-guide-to-algorithmic-strategies-and-trading-systems.pdf',
                'DiGA for LOB.pdf',
                'Using_Directional_Change_for_Information_Extraction_in_Financial_Market_Data_-_Tao_Ran.pdf'
            ],
            'research_specialized': [
                'topological-data-analysis-for-scientific-visualization.pdf',
                'dp439.pdf'
            ]
        }

    async def process_book_collection(self, books_directory: Path) -> dict[str, BookAnalysis]:
        """Process entire book collection in phases"""
        logger.info("🚀 Starting comprehensive book processing pipeline")
        logger.info("=" * 80)

        all_analyses = {}
        knowledge_connections = []

        for phase_name, book_files in self.processing_phases.items():
            logger.info(f"\n📚 Processing Phase: {phase_name.upper()}")
            logger.info("-" * 60)

            phase_analyses = await self._process_phase(books_directory, book_files, phase_name)
            all_analyses.update(phase_analyses)

            # Build knowledge connections within phase
            phase_connections = await self._build_phase_connections(phase_analyses)
            knowledge_connections.extend(phase_connections)

        # Build cross-phase knowledge synthesis
        cross_phase_synthesis = await self._build_cross_phase_synthesis(all_analyses)

        # Generate comprehensive knowledge map
        await self._generate_knowledge_map(all_analyses, knowledge_connections, cross_phase_synthesis)

        logger.info(f"\n✅ Completed processing {len(all_analyses)} books")
        return all_analyses

    async def _process_phase(self, books_dir: Path, book_files: list[str], phase_name: str) -> dict[str, BookAnalysis]:
        """Process books in a specific phase"""
        phase_analyses = {}

        for book_file in book_files:
            book_path = books_dir / book_file

            if not book_path.exists():
                logger.warning(f"Book not found: {book_file}")
                continue

            logger.info(f"\n📖 Processing: {book_file}")

            try:
                analysis = await self.process_single_book(book_path, phase_name)
                phase_analyses[book_file] = analysis

                # Save individual analysis
                await self._save_book_analysis(analysis)

            except Exception as e:
                logger.error(f"Error processing {book_file}: {e}")

        return phase_analyses

    async def process_single_book(self, book_path: Path, knowledge_level: str) -> BookAnalysis:
        """Process a single book with comprehensive analysis"""
        logger.info(f"📄 Extracting content from {book_path.name}...")

        # Extract book content
        content = await self._extract_book_content(book_path)
        total_pages = await self._get_page_count(book_path)

        # Compress content for efficient processing
        logger.info("🗜️  Compressing content for analysis...")
        compressed_chunks = await self._compress_book_content(content, book_path.name)

        # Extract different types of knowledge
        logger.info("🧠 Extracting conceptual frameworks...")
        frameworks = await self._extract_conceptual_frameworks(compressed_chunks, book_path.name)

        logger.info("📊 Extracting trading strategies...")
        strategies = await self._extract_trading_strategies(compressed_chunks, book_path.name)

        logger.info("💡 Extracting key insights...")
        insights = await self._extract_key_insights(compressed_chunks)

        logger.info("🔢 Extracting mathematical content...")
        math_content = await self._extract_mathematical_content(compressed_chunks)

        logger.info("💻 Extracting implementation examples...")
        implementations = await self._extract_implementation_examples(compressed_chunks)

        logger.info("📈 Identifying data requirements...")
        data_reqs = await self._identify_data_requirements(compressed_chunks)

        # Calculate scores
        practical_score = await self._calculate_practical_value(frameworks, strategies, implementations)
        implementation_score = await self._calculate_implementation_readiness(strategies, implementations)

        # Build analysis result
        analysis = BookAnalysis(
            book_title=book_path.stem,
            book_path=str(book_path),
            processing_date=datetime.now(),
            total_pages=total_pages,
            conceptual_frameworks=frameworks,
            trading_strategies=strategies,
            key_insights=insights,
            mathematical_content=math_content,
            implementation_examples=implementations,
            data_requirements=data_reqs,
            cross_references=[],  # Will be populated later
            knowledge_level=knowledge_level,
            practical_value_score=practical_score,
            implementation_readiness=implementation_score
        )

        logger.info(f"✅ Completed analysis: {len(frameworks)} frameworks, {len(strategies)} strategies")
        return analysis

    async def _extract_book_content(self, book_path: Path) -> str:
        """Extract full text content from book PDF"""
        try:
            doc = fitz.open(book_path)
            full_text = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"

            doc.close()
            return full_text

        except Exception as e:
            logger.error(f"Error extracting text from {book_path}: {e}")
            return ""

    async def _get_page_count(self, book_path: Path) -> int:
        """Get total page count of the book"""
        try:
            doc = fitz.open(book_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Could not get page count for {book_path}: {e}")
            return 0

    async def _compress_book_content(self, content: str, book_title: str) -> list[CompressionResult]:
        """
        Compress book content with quality preservation for trading knowledge extraction.

        This is CRITICAL for financial applications - no information loss that could
        affect trading strategy validity or mathematical formulas.
        """
        # Split content into logical chunks (by sections or page ranges)
        chunks = self._split_into_logical_chunks(content)
        logger.info(f"Split {book_title} into {len(chunks)} chunks for processing")

        compressed_results = []
        for i, chunk_content in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {book_title}")

            # Create proper Chunk model with compatibility adapter
            chunk = self._create_chunk_adapter(chunk_content, book_title, i)

            try:
                # Use proper compression with mathematical content preservation
                result = await self.compressor.compress_chunk(chunk, preserve_math=True)

                # Validate compression quality for financial content
                if self.validation_enabled:
                    validation_result = self._validate_compression_quality(chunk_content, result)
                    if not validation_result.is_valid:
                        logger.warning(f"Compression quality issue in chunk {i}: {validation_result.issues}")
                        # Use original text if compression compromises quality
                        result = CompressionResult(
                            original_text=chunk_content,
                            compressed_text=chunk_content,
                            compression_ratio=1.0,
                            tokens_saved=0,
                            quality_score=1.0
                        )

                compressed_results.append(result)

            except Exception as e:
                logger.error(f"Compression failed for chunk {i} in {book_title}: {e}")
                # Fallback to original content - no data loss
                result = CompressionResult(
                    original_text=chunk_content,
                    compressed_text=chunk_content,
                    compression_ratio=1.0,
                    tokens_saved=0,
                    quality_score=1.0
                )
                compressed_results.append(result)

        logger.info(f"Compressed {len(compressed_results)} chunks for {book_title}")
        return compressed_results

    def _split_into_logical_chunks(self, content: str, chunk_size: int = 5000) -> list[str]:
        """Split content into logical chunks for processing"""
        # Simple chunking by character count with sentence boundary preservation
        chunks = []
        current_chunk = ""

        sentences = content.split('. ')

        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _create_chunk_adapter(self, content: str, book_title: str, chunk_index: int):
        """Create Chunk object compatible with compressor"""
        # Create a simple object that has the 'content' attribute expected by compressor
        class ChunkAdapter:
            def __init__(self, text_content):
                self.content = text_content  # Compressor expects 'content'
                self.text = text_content     # Our model uses 'text'

        return ChunkAdapter(content)

    def _validate_compression_quality(self, original: str, compressed_result: CompressionResult):
        """
        Validate compression quality for financial content.

        Trading strategies and mathematical formulas MUST be preserved exactly.
        Better to use uncompressed text than lose critical information.
        """
        class ValidationResult:
            def __init__(self):
                self.is_valid = True
                self.issues = []

        result = ValidationResult()

        # Check for critical financial terms preservation
        financial_terms = [
            'sharpe', 'volatility', 'correlation', 'regression', 'beta', 'alpha',
            'return', 'profit', 'loss', 'risk', 'portfolio', 'asset', 'strategy',
            'signal', 'entry', 'exit', 'stop', 'limit', 'price', 'volume'
        ]

        for term in financial_terms:
            if term in original.lower() and term not in compressed_result.compressed_text.lower():
                result.is_valid = False
                result.issues.append(f"Lost critical financial term: {term}")

        # Check for mathematical expressions preservation
        math_patterns = [
            r'\$[^$]+\$',  # LaTeX math
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'\d+\.\d+',  # Decimal numbers
            r'[=<>]+',  # Mathematical operators
            r'[+\-*/]',  # Arithmetic operators
        ]

        for pattern in math_patterns:
            original_matches = len(re.findall(pattern, original))
            compressed_matches = len(re.findall(pattern, compressed_result.compressed_text))
            if original_matches > compressed_matches:
                result.is_valid = False
                result.issues.append(f"Lost mathematical expressions: {pattern}")

        # Check compression ratio isn't too aggressive
        if compressed_result.compression_ratio < 0.3:  # More than 70% compression is suspicious
            result.is_valid = False
            result.issues.append(f"Compression too aggressive: {compressed_result.compression_ratio}")

        return result

    def _validate_trading_strategy(self, strategy: TradingStrategy):
        """
        Validate trading strategy for completeness and financial soundness.

        Trading strategies MUST have:
        - Clear entry and exit criteria
        - Risk management rules
        - Testable validation method
        - Reasonable performance expectations
        """
        class ValidationResult:
            def __init__(self):
                self.is_valid = True
                self.issues = []

        result = ValidationResult()

        # Check required fields are not empty
        required_fields = {
            'name': strategy.name,
            'entry_criteria': strategy.entry_criteria,
            'exit_criteria': strategy.exit_criteria,
            'risk_management': strategy.risk_management,
            'validation_method': strategy.validation_method
        }

        for field_name, field_value in required_fields.items():
            if not field_value or field_value.strip() == '':
                result.is_valid = False
                result.issues.append(f"Missing required field: {field_name}")

        # Validate entry criteria specificity
        entry_vague_terms = ['when appropriate', 'if profitable', 'good opportunity']
        if any(term in strategy.entry_criteria.lower() for term in entry_vague_terms):
            result.is_valid = False
            result.issues.append("Entry criteria too vague - needs specific technical conditions")

        # Validate exit criteria specificity
        exit_vague_terms = ['when appropriate', 'take profit', 'cut losses']
        if any(term in strategy.exit_criteria.lower() for term in exit_vague_terms):
            result.is_valid = False
            result.issues.append("Exit criteria too vague - needs specific price/indicator targets")

        # Check for risk management specifics
        risk_keywords = ['stop', 'position size', 'risk', 'drawdown', 'limit']
        if not any(keyword in strategy.risk_management.lower() for keyword in risk_keywords):
            result.is_valid = False
            result.issues.append("Risk management lacks specific controls")

        # Validate performance expectations are reasonable
        if 'guaranteed' in strategy.expected_performance.lower():
            result.is_valid = False
            result.issues.append("No trading strategy can guarantee profits")

        # Check for unrealistic returns
        unrealistic_claims = ['100%', '1000%', 'never lose', 'always win', 'risk-free']
        if any(claim in strategy.expected_performance.lower() for claim in unrealistic_claims):
            result.is_valid = False
            result.issues.append("Unrealistic performance expectations")

        return result

    async def _extract_conceptual_frameworks(self, compressed_chunks: list[CompressionResult], book_source: str) -> list[ConceptualFramework]:
        """Extract conceptual frameworks using Qwen2.5-Coder"""
        frameworks = []

        for chunk in compressed_chunks[:3]:  # Process first few chunks for frameworks
            prompt = f"""Analyze this trading/finance book content and extract key conceptual frameworks:

{chunk.compressed_text}

Extract frameworks in this JSON format:
{{
  "frameworks": [
    {{
      "name": "Framework Name",
      "description": "Clear description of the framework",
      "mathematical_foundation": "Mathematical basis (if any)",
      "implementation_approach": "How to implement this framework",
      "prerequisites": ["prerequisite1", "prerequisite2"],
      "applications": ["application1", "application2"],
      "complexity_level": "beginner|intermediate|advanced",
      "testable_hypothesis": "Specific testable hypothesis (if applicable)"
    }}
  ]
}}

Focus on actionable frameworks that can be implemented and tested."""

            try:
                response = await self._query_qwen(prompt)
                frameworks_data = self._parse_json_response(response)

                if frameworks_data and 'frameworks' in frameworks_data:
                    for fw_data in frameworks_data['frameworks']:
                        framework = ConceptualFramework(
                            name=fw_data.get('name', ''),
                            description=fw_data.get('description', ''),
                            mathematical_foundation=fw_data.get('mathematical_foundation', ''),
                            implementation_approach=fw_data.get('implementation_approach', ''),
                            prerequisites=fw_data.get('prerequisites', []),
                            applications=fw_data.get('applications', []),
                            book_source=book_source,
                            page_references=[],  # Would need position mapping
                            complexity_level=fw_data.get('complexity_level', 'intermediate'),
                            testable_hypothesis=fw_data.get('testable_hypothesis')
                        )
                        frameworks.append(framework)

            except Exception as e:
                logger.error(f"Error extracting frameworks: {e}")

        return frameworks

    async def _extract_trading_strategies(self, compressed_chunks: list[CompressionResult], book_source: str) -> list[TradingStrategy]:
        """Extract trading strategies using Qwen2.5-Coder"""
        strategies = []

        for chunk in compressed_chunks:
            prompt = f"""Analyze this trading/finance content and extract specific trading strategies:

{chunk.compressed_text}

Extract strategies in this JSON format:
{{
  "strategies": [
    {{
      "name": "Strategy Name",
      "description": "Clear strategy description",
      "entry_criteria": "Specific entry conditions",
      "exit_criteria": "Specific exit conditions",
      "risk_management": "Risk management approach",
      "time_horizon": "short-term|medium-term|long-term",
      "asset_classes": ["stocks", "futures", "crypto", "etc"],
      "data_requirements": ["required data types"],
      "expected_performance": "Performance characteristics",
      "implementation_complexity": "low|medium|high",
      "validation_method": "How to test this strategy"
    }}
  ]
}}

Focus on specific, implementable strategies with clear rules."""

            try:
                response = await self._query_qwen(prompt)
                strategies_data = self._parse_json_response(response)

                if strategies_data and 'strategies' in strategies_data:
                    for strat_data in strategies_data['strategies']:
                        strategy = TradingStrategy(
                            name=strat_data.get('name', ''),
                            description=strat_data.get('description', ''),
                            entry_criteria=strat_data.get('entry_criteria', ''),
                            exit_criteria=strat_data.get('exit_criteria', ''),
                            risk_management=strat_data.get('risk_management', ''),
                            time_horizon=strat_data.get('time_horizon', 'medium-term'),
                            asset_classes=strat_data.get('asset_classes', []),
                            data_requirements=strat_data.get('data_requirements', []),
                            expected_performance=strat_data.get('expected_performance', ''),
                            implementation_complexity=strat_data.get('implementation_complexity', 'medium'),
                            book_source=book_source,
                            validation_method=strat_data.get('validation_method', '')
                        )

                        # Validate strategy completeness and accuracy
                        if self.validation_enabled:
                            validation_result = self._validate_trading_strategy(strategy)
                            if validation_result.is_valid:
                                strategies.append(strategy)
                            else:
                                logger.warning(f"Strategy validation failed: {strategy.name} - {validation_result.issues}")
                        else:
                            strategies.append(strategy)

            except Exception as e:
                logger.error(f"Error extracting strategies: {e}")

        return strategies

    async def _extract_key_insights(self, compressed_chunks: list[CompressionResult]) -> list[str]:
        """Extract key insights and takeaways"""
        insights = []

        for chunk in compressed_chunks[:2]:  # First couple chunks for main insights
            prompt = f"""Extract the 5 most important insights from this trading/finance content:

{chunk.compressed_text}

Provide insights as a JSON array of strings:
{{
  "insights": [
    "Insight 1: Clear, actionable insight",
    "Insight 2: Another key takeaway"
  ]
}}

Focus on actionable insights that could impact trading decisions."""

            try:
                response = await self._query_qwen(prompt)
                insights_data = self._parse_json_response(response)

                if insights_data and 'insights' in insights_data:
                    insights.extend(insights_data['insights'])

            except Exception as e:
                logger.error(f"Error extracting insights: {e}")

        return insights[:10]  # Limit to top 10

    async def _extract_mathematical_content(self, compressed_chunks: list[CompressionResult]) -> list[str]:
        """Extract mathematical formulas and models"""
        math_content = []

        for chunk in compressed_chunks:
            # Look for mathematical expressions in compressed text
            if any(indicator in chunk.compressed_text.lower() for indicator in
                   ['formula', 'equation', 'model', 'algorithm', 'calculation']):

                prompt = f"""Extract mathematical formulas, equations, and models from this content:

{chunk.compressed_text}

Provide mathematical content as JSON:
{{
  "mathematical_content": [
    "Formula/Model 1: Description and mathematical expression",
    "Formula/Model 2: Another mathematical concept"
  ]
}}

Focus on formulas that are relevant to trading and finance."""

                try:
                    response = await self._query_qwen(prompt)
                    math_data = self._parse_json_response(response)

                    if math_data and 'mathematical_content' in math_data:
                        # Validate mathematical content before adding
                        for formula in math_data['mathematical_content']:
                            if self._validate_mathematical_formula(formula):
                                math_content.append(formula)
                            else:
                                logger.warning(f"Invalid mathematical formula rejected: {formula[:100]}...")

                except Exception as e:
                    logger.error(f"Error extracting mathematical content: {e}")

        return math_content

    async def _extract_implementation_examples(self, compressed_chunks: list[CompressionResult]) -> list[str]:
        """Extract code examples and implementation patterns"""
        implementations = []

        for chunk in compressed_chunks:
            if any(indicator in chunk.compressed_text.lower() for indicator in
                   ['python', 'code', 'implementation', 'algorithm', 'function']):

                prompt = f"""Extract implementation examples and code patterns from this content:

{chunk.compressed_text}

Provide implementations as JSON:
{{
  "implementations": [
    "Implementation 1: Description and code approach",
    "Implementation 2: Another implementation pattern"
  ]
}}

Focus on practical implementation approaches for trading systems."""

                try:
                    response = await self._query_qwen(prompt)
                    impl_data = self._parse_json_response(response)

                    if impl_data and 'implementations' in impl_data:
                        implementations.extend(impl_data['implementations'])

                except Exception as e:
                    logger.error(f"Error extracting implementations: {e}")

        return implementations

    async def _identify_data_requirements(self, compressed_chunks: list[CompressionResult]) -> list[str]:
        """Identify data requirements for strategies and frameworks"""
        data_reqs = set()

        for chunk in compressed_chunks:
            # Common data requirement patterns
            data_patterns = [
                r'\b(price|OHLC|tick|minute|daily)\s+data\b',
                r'\b(volume|liquidity|order book)\s+data\b',
                r'\b(fundamental|earnings|financial)\s+data\b',
                r'\b(market|economic|macro)\s+data\b',
                r'\b(news|sentiment|social)\s+data\b'
            ]

            for pattern in data_patterns:
                matches = re.findall(pattern, chunk.compressed_text, re.IGNORECASE)
                for match in matches:
                    data_reqs.add(match.strip())

        return list(data_reqs)

    async def _calculate_practical_value(self, frameworks: list[ConceptualFramework],
                                        strategies: list[TradingStrategy],
                                        implementations: list[str]) -> float:
        """Calculate practical value score for the book"""
        score = 0.0

        # Weight different components
        score += len(strategies) * 0.3  # Strategies are high value
        score += len(frameworks) * 0.2  # Frameworks provide foundation
        score += len(implementations) * 0.25  # Implementation examples are valuable

        # Bonus for specific, actionable content
        for strategy in strategies:
            if strategy.entry_criteria and strategy.exit_criteria:
                score += 0.1
            if strategy.data_requirements:
                score += 0.05

        return min(score, 1.0)  # Cap at 1.0

    async def _calculate_implementation_readiness(self, strategies: list[TradingStrategy],
                                                implementations: list[str]) -> float:
        """Calculate how ready the content is for implementation"""
        score = 0.0

        # Check for implementation details
        score += len(implementations) * 0.2

        for strategy in strategies:
            if strategy.implementation_complexity == 'low':
                score += 0.15
            elif strategy.implementation_complexity == 'medium':
                score += 0.1

            if strategy.validation_method:
                score += 0.1

        return min(score, 1.0)

    async def _build_phase_connections(self, phase_analyses: dict[str, BookAnalysis]) -> list[dict]:
        """Build knowledge connections within a processing phase"""
        connections = []

        # Find overlapping concepts and strategies
        all_frameworks = []
        all_strategies = []

        for analysis in phase_analyses.values():
            all_frameworks.extend(analysis.conceptual_frameworks)
            all_strategies.extend(analysis.trading_strategies)

        # Simple similarity matching (could be enhanced)
        for i, fw1 in enumerate(all_frameworks):
            for fw2 in all_frameworks[i+1:]:
                if self._concepts_related(fw1.name, fw2.name):
                    connections.append({
                        'type': 'conceptual_overlap',
                        'source': fw1.book_source,
                        'target': fw2.book_source,
                        'concept1': fw1.name,
                        'concept2': fw2.name
                    })

        return connections

    def _concepts_related(self, concept1: str, concept2: str) -> bool:
        """Simple concept similarity check"""
        # Basic keyword overlap
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())
        overlap = len(words1.intersection(words2))
        return overlap > 0 and overlap / min(len(words1), len(words2)) > 0.3

    async def _build_cross_phase_synthesis(self, all_analyses: dict[str, BookAnalysis]) -> dict:
        """Build synthesis across all processing phases"""
        synthesis = {
            'foundational_concepts': [],
            'implementation_patterns': [],
            'data_infrastructure_needs': [],
            'testing_frameworks': [],
            'performance_expectations': []
        }

        # Aggregate insights across all books
        for analysis in all_analyses.values():
            # Extract foundational concepts
            for framework in analysis.conceptual_frameworks:
                if framework.complexity_level == 'beginner':
                    synthesis['foundational_concepts'].append({
                        'concept': framework.name,
                        'source': analysis.book_title
                    })

            # Aggregate data requirements
            synthesis['data_infrastructure_needs'].extend(analysis.data_requirements)

        # Remove duplicates
        synthesis['data_infrastructure_needs'] = list(set(synthesis['data_infrastructure_needs']))

        return synthesis

    async def _generate_knowledge_map(self, all_analyses: dict[str, BookAnalysis],
                                    connections: list[dict], synthesis: dict):
        """Generate comprehensive knowledge map"""
        knowledge_map = {
            'processing_summary': {
                'total_books': len(all_analyses),
                'total_frameworks': sum(len(a.conceptual_frameworks) for a in all_analyses.values()),
                'total_strategies': sum(len(a.trading_strategies) for a in all_analyses.values()),
                'processing_date': datetime.now().isoformat()
            },
            'book_analyses': {title: self._serialize_analysis(analysis)
                            for title, analysis in all_analyses.items()},
            'knowledge_connections': connections,
            'cross_phase_synthesis': synthesis
        }

        # Save comprehensive knowledge map
        output_file = Path("/home/scott/TradeKnowledge/comprehensive_knowledge_map.json")
        with open(output_file, 'w') as f:
            json.dump(knowledge_map, f, indent=2, default=str)

        logger.info(f"📊 Knowledge map saved to: {output_file}")

    def _serialize_analysis(self, analysis: BookAnalysis) -> dict:
        """Convert BookAnalysis to JSON-serializable dict"""
        return {
            'book_title': analysis.book_title,
            'processing_date': analysis.processing_date.isoformat(),
            'total_pages': analysis.total_pages,
            'knowledge_level': analysis.knowledge_level,
            'practical_value_score': analysis.practical_value_score,
            'implementation_readiness': analysis.implementation_readiness,
            'framework_count': len(analysis.conceptual_frameworks),
            'strategy_count': len(analysis.trading_strategies),
            'key_insights_count': len(analysis.key_insights),
            'data_requirements': analysis.data_requirements,
            'frameworks': [{
                'name': fw.name,
                'description': fw.description,
                'complexity_level': fw.complexity_level,
                'testable_hypothesis': fw.testable_hypothesis
            } for fw in analysis.conceptual_frameworks],
            'strategies': [{
                'name': strat.name,
                'description': strat.description,
                'time_horizon': strat.time_horizon,
                'asset_classes': strat.asset_classes,
                'implementation_complexity': strat.implementation_complexity
            } for strat in analysis.trading_strategies]
        }

    async def _save_book_analysis(self, analysis: BookAnalysis):
        """Save individual book analysis"""
        filename = f"analysis_{analysis.book_title.replace(' ', '_')}.json"
        output_file = Path(f"/home/scott/TradeKnowledge/book_analyses/{filename}")
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self._serialize_analysis(analysis), f, indent=2, default=str)

    async def _query_qwen(self, prompt: str) -> str:
        """
        Query Qwen2.5-Coder model via Ollama with retry mechanism.

        Financial analysis requires reliable LLM responses. We retry with
        exponential backoff to ensure data quality over speed.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,  # Very low temperature for consistency
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 8192  # Larger context for complex financial content
            }
        }

        for attempt in range(self.max_retries):
            try:
                timeout = 180 + (attempt * 60)  # Increase timeout with retries
                response = requests.post(self.ollama_url, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()

                response_text = result.get('response', '').strip()

                # Validate response quality
                if self._validate_llm_response(response_text, prompt):
                    return response_text
                else:
                    logger.warning(f"LLM response quality insufficient, attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue

            except requests.exceptions.Timeout:
                logger.warning(f"LLM timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as e:
                logger.error(f"Error querying Qwen (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue

        logger.error(f"Failed to get quality response after {self.max_retries} attempts")
        return ""

    def _validate_llm_response(self, response: str, prompt: str) -> bool:
        """
        Validate LLM response quality for financial content extraction.

        Ensures responses are complete, relevant, and contain expected structure.
        """
        if not response or len(response.strip()) < 10:
            return False

        # Check for JSON structure if expected
        if 'JSON' in prompt.upper() or 'json' in prompt:
            try:
                # Try to find and parse JSON in response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    json.loads(json_str)
                    return True
                else:
                    return False
            except json.JSONDecodeError:
                return False

        # Check for financial relevance
        financial_indicators = [
            'trading', 'strategy', 'portfolio', 'risk', 'return', 'market',
            'price', 'volume', 'analysis', 'algorithm', 'indicator'
        ]

        if not any(indicator in response.lower() for indicator in financial_indicators):
            return False

        return True

    def _validate_mathematical_formula(self, formula: str) -> bool:
        """
        Validate mathematical formulas for accuracy and completeness.

        Mathematical errors in trading can lead to significant financial losses.
        We validate formulas for basic mathematical soundness.
        """
        if not formula or len(formula.strip()) < 5:
            return False

        # Check for basic mathematical indicators
        math_indicators = [
            '=', '+', '-', '*', '/', '^', 'sqrt', 'log', 'exp', 'sin', 'cos',
            'sum', 'mean', 'std', 'var', 'correlation', 'beta', 'alpha'
        ]

        if not any(indicator in formula.lower() for indicator in math_indicators):
            return False

        # Check for financial relevance
        financial_math = [
            'return', 'volatility', 'sharpe', 'sortino', 'risk', 'portfolio',
            'price', 'volume', 'correlation', 'covariance', 'beta', 'alpha',
            'drawdown', 'var', 'cvar', 'expected', 'probability'
        ]

        if not any(term in formula.lower() for term in financial_math):
            return False

        # Check for obvious errors
        error_patterns = [
            r'divide by zero', r'/\s*0\s*[^.]', r'undefined', r'infinity',
            r'error', r'invalid', r'null', r'nan'
        ]

        for pattern in error_patterns:
            if re.search(pattern, formula.lower()):
                return False

        return True

    def _parse_json_response(self, response: str) -> dict | None:
        """Parse JSON response from Qwen"""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Error parsing JSON response: {e}")

        return None


async def main():
    """Main function to run comprehensive book processing"""
    processor = ComprehensiveBookProcessor()
    books_directory = Path("/home/scott/TradeKnowledge/books and papers (pdf and epub)")

    if not books_directory.exists():
        logger.error(f"Books directory not found: {books_directory}")
        return

    # Process all books
    analyses = await processor.process_book_collection(books_directory)

    # Print summary
    print("\n" + "="*80)
    print("📚 COMPREHENSIVE BOOK PROCESSING COMPLETE")
    print("="*80)

    total_frameworks = sum(len(a.conceptual_frameworks) for a in analyses.values())
    total_strategies = sum(len(a.trading_strategies) for a in analyses.values())

    print(f"📖 Books processed: {len(analyses)}")
    print(f"🧠 Conceptual frameworks extracted: {total_frameworks}")
    print(f"📊 Trading strategies identified: {total_strategies}")
    print("💾 Knowledge map saved: comprehensive_knowledge_map.json")

    # Show top insights
    print("\n🔝 Top Practical Insights:")
    for _title, analysis in sorted(analyses.items(), key=lambda x: x[1].practical_value_score, reverse=True)[:3]:
        print(f"  • {analysis.book_title}: {analysis.practical_value_score:.2f} practical value")
        print(f"    {len(analysis.trading_strategies)} strategies, {len(analysis.conceptual_frameworks)} frameworks")


if __name__ == "__main__":
    asyncio.run(main())
