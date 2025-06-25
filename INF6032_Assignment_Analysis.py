# Databricks notebook source
# Q1: “Calculate the number of different sentences in the dataset.”
# Step 1: Load Wikipedia data (large.csv.gz)
wiki_df = spark.read.csv("/FileStore/tables/large_csv.gz", header=True, inferSchema=True)
# Check the first few lines to confirm the format
wiki_df.show(5)
# Clean up empty lines or invalid sentences
wiki_df = wiki_df.filter(wiki_df["sentence"].isNotNull() & (wiki_df["sentence"] != ""))
# Step 2: Select the sentence column + remove duplicates
distinct_sentences = wiki_df.select("sentence").distinct()
# Step 3: Count the number of different sentences
num_distinct_sentences = distinct_sentences.count()
# Step 4: Output the results
print(f"Number of distinct sentences in large.csv.gz: {num_distinct_sentences}")


# Q2: “Longest Sentences by Word Count.”
from pyspark.sql.functions import split, size
# Count the words
wiki_with_word_count = wiki_df.withColumn("word_count", size(split(wiki_df["sentence"], " ")))
# Get the number of words in the top 10 longest sentences
top10_word_counts = wiki_with_word_count.orderBy("word_count", ascending=False).select("word_count").limit(10)
top10_word_counts.show()

# Q3: “Calculate the average number of bigrams per sentence.”
from pyspark.sql.functions import split, size, when, col, sum as spark_sum
# Step 1: Loading data + cleaning
wiki_df = spark.read.csv("/FileStore/tables/large_csv.gz", header=True, inferSchema=True)
wiki_df = wiki_df.filter(wiki_df["sentence"].isNotNull() & (wiki_df["sentence"] != ""))
# Step 2: Tokenize and count words
wiki_df = wiki_df.withColumn("word_count", size(split(col("sentence"), " ")))
# Step 3: Calculate the number of bigrams per sentence
wiki_df = wiki_df.withColumn("bigram_count", when(col("word_count") >= 2, col("word_count") - 1).otherwise(0))
# Step 4: Find the total number of bigrams
total_bigrams = wiki_df.agg(spark_sum("bigram_count")).collect()[0][0]
# Step 5: Find the total number of sentences
total_sentences = wiki_df.count()
# Step 6: average number of bigrams
avg_bigrams = total_bigrams / total_sentences
print(f"Average number of bigrams per sentence: {avg_bigrams:.4f}")



# Q4: “Calculate the average number of bigrams per sentence.”
from pyspark.sql.functions import split, regexp_replace, explode, col
from pyspark.ml.feature import NGram
# Step 1: Loading and cleaning data
wiki_df = spark.read.csv("/FileStore/tables/large_csv.gz", header=True, inferSchema=True)
wiki_df = wiki_df.filter(wiki_df["sentence"].isNotNull() & (wiki_df["sentence"] != ""))
# Step 2: Remove punctuation (keep only letters, numbers, and spaces)
wiki_clean = wiki_df.withColumn("clean_sentence", regexp_replace(col("sentence"), r"[^\w\s]", ""))
# Step 3: Word segmentation (based on clean_sentence）
wiki_tokenized = wiki_clean.withColumn("words", split(col("clean_sentence"), " "))
# Step 4: Generating bigrams using NGram
ngram = NGram(n=2, inputCol="words", outputCol="bigrams")
wiki_bigrams_df = ngram.transform(wiki_tokenized)
# Step 5: Expand bigrams into one row per column
exploded_bigrams = wiki_bigrams_df.select(explode(col("bigrams")).alias("bigram"))
# Step 6: Count frequencies & get the most common bigrams
most_common_bigram = exploded_bigrams.groupBy("bigram").count().orderBy(col("count").desc()).limit(1)
# Step 7: Output
most_common_bigram.show(truncate=False)


# Q5: “How many idioms occur in the Wikipedia data?”
# Find the maximum number of words in the MAGPIE dataset
from pyspark.sql.functions import size, split
# Extract the idiom field from the MAGPIE dataset
magpie = spark.read.json("/FileStore/tables/MAGPIE_unfiltered.jsonl")
idioms = magpie.select("idiom").dropna().dropDuplicates()
# Remove punctuation
from pyspark.sql.functions import regexp_replace
idioms = idioms.withColumn("clean_idiom", regexp_replace("idiom", r"[^\w\s]", ""))
# Split idiom into a list of words
idioms = idioms.withColumn("word_count", size(split("clean_idiom", " ")))
# Find the maximum number of words
max_len = idioms.agg({"word_count": "max"}).collect()[0][0]
print(f"Maximum number of words in any idiom: {max_len}")


# MAIN BODY
from pyspark.sql.functions import col, split, explode, regexp_replace, lower
from pyspark.ml.feature import NGram
# Step 1: Loading and cleaning Wikipedia sentences
wiki_df = spark.read.csv("/FileStore/tables/large_csv.gz", header=True, inferSchema=True)
wiki_df = wiki_df.filter(wiki_df["sentence"].isNotNull() & (wiki_df["sentence"] != ""))
wiki_df = wiki_df.withColumn("clean_sentence", regexp_replace(col("sentence"), r"[^\w\s]", ""))
wiki_df = wiki_df.withColumn("words", split(col("clean_sentence"), " "))
# Step 2: Load and clean the MAGPIE idioms
magpie_df = spark.read.json("/FileStore/tables/MAGPIE_unfiltered.jsonl")
idioms = magpie_df.select("idiom").dropna().dropDuplicates()
idioms = idioms.withColumn("idiom_clean", regexp_replace(col("idiom"), r"[^\w\s]", ""))
idioms = idioms.withColumn("idiom_clean", lower(col("idiom_clean")))
# Step 3: Matches all idioms from 2 to 9-grams
matched_ngrams = None

for n in range(2, 10):  # From 2 to 9
    ngrammer = NGram(n=n, inputCol="words", outputCol="ngrams")
    ngram_df = ngrammer.transform(wiki_df)
    exploded = ngram_df.select(explode(col("ngrams")).alias("ngram"))
    
    # Unified format (remove punctuation, convert to lowercase)
    exploded = exploded.withColumn("ngram", lower(regexp_replace(col("ngram"), r"[^\w\s]", "")))
    
    # Inner join matching idiom
    matched = exploded.join(idioms, exploded["ngram"] == idioms["idiom_clean"])
    
    # Merge multiple matches of n
    matched_ngrams = matched_ngrams.union(matched) if matched_ngrams else matched

# Step 4: Count the number of idioms that appear (remove duplicates)
idiom_count = matched_ngrams.select("idiom_clean").distinct().count()
print(f"Number of idioms from MAGPIE found in Wikipedia: {idiom_count}")

# Q6: ““Print out the 10 bigrams starting from rank 2500 (ranked by frequency descending), skipping any that appear in MAGPIE. For ties, use alphabetical order.”
from pyspark.sql.functions import split, regexp_replace, explode, col, lower, count as spark_count
from pyspark.ml.feature import NGram

# Step 1: Loading and cleaning Wikipedia sentences
wiki_df = spark.read.csv("/FileStore/tables/large_csv.gz", header=True, inferSchema=True)
wiki_df = wiki_df.filter(wiki_df["sentence"].isNotNull() & (wiki_df["sentence"] != ""))
wiki_df = wiki_df.withColumn("clean_sentence", regexp_replace(col("sentence"), r"[^\w\s]", ""))
wiki_df = wiki_df.withColumn("words", split(col("clean_sentence"), " "))

# Step 2: Generate bigrams from Wikipedia
ngrammer = NGram(n=2, inputCol="words", outputCol="bigrams")
wiki_bigrams_df = ngrammer.transform(wiki_df)
wiki_bigrams = wiki_bigrams_df.select(explode(col("bigrams")).alias("bigram"))
wiki_bigrams = wiki_bigrams.withColumn("bigram", lower(regexp_replace(col("bigram"), r"[^\w\s]", "")))

# Step 3: Count the frequency of bigram occurrences
wiki_bigram_freq = wiki_bigrams.groupBy("bigram").agg(spark_count("*").alias("freq"))

# Step 4: Extract all bigrams in MAGPIE idioms (only generate n=2)
magpie = spark.read.json("/FileStore/tables/MAGPIE_unfiltered.jsonl")
idioms = magpie.select("idiom").dropna().dropDuplicates()
idioms = idioms.withColumn("clean_idiom", regexp_replace(lower(col("idiom")), r"[^\w\s]", ""))
idioms = idioms.withColumn("words", split(col("clean_idiom"), " "))
magpie_bigrams = NGram(n=2, inputCol="words", outputCol="bigrams").transform(idioms)
magpie_bigram_list = magpie_bigrams.select(explode(col("bigrams")).alias("magpie_bigram"))
magpie_bigram_list = magpie_bigram_list.withColumn("magpie_bigram", lower(regexp_replace(col("magpie_bigram"), r"[^\w\s]", "")))

# Step 5: Remove bigrams from MAGPIE
filtered_wiki_bigrams = wiki_bigram_freq.join(
    magpie_bigram_list,
    wiki_bigram_freq["bigram"] == magpie_bigram_list["magpie_bigram"],
    how="left_anti"  # 只保留 wiki 中不在 MAGPIE 中的 bigram
)

# Step 6: Sort and take the 10 items ranked 2500~2509
# (Frequency descending + alphabetical ascending)
result = filtered_wiki_bigrams.orderBy(col("freq").desc(), col("bigram").asc()).limit(2510).tail(10)

# Step 7: Printing Results
print("Top bigrams ranked 2500 to 2509 (excluding MAGPIE idioms):")
for row in result:
    print(f"{row['bigram']} ({row['freq']})")



