#------------------------#
# Configurable variables #
#------------------------#

# Delete files on error or iterruption (does not delete folder targets)
.DELETE_ON_ERROR:

#------#
# Data #
#------#

STUDIES := Galson_2015a Galson_2016
DATA_PATH := /SFS/user/wp/prihodad/git/oas-dataset/data
DATAUNIT_FILE_EXT := .json.gz
UNITS_LIST_FILE_EXT := .txt
PARQUET_FILE_EXT := .parquet
BUILD_DATAUNIT_SCRIPT := bin/build_dataunit.py
BUILD_COMBINED_DATA_SCRIPT := bin/build_combined_data.py

data: data/seq

data/seq: $(patsubst %,data/seq/%,$(STUDIES))

data/seq/%: data/meta/units-list/%$(UNITS_LIST_FILE_EXT)
	make $(patsubst %,$@/%$(PARQUET_FILE_EXT),$(shell cat "$<"))

data/meta/units-list/%$(UNITS_LIST_FILE_EXT): $(DATA_PATH)/human/meta/units-list/%$(UNITS_LIST_FILE_EXT)
	mkdir -p $(@D)
	ln -s "$<" "$@"

data/seq/%$(PARQUET_FILE_EXT): $(DATA_PATH)/all/json/%$(DATAUNIT_FILE_EXT)
	mkdir -p $(@D)
	hpc/conda-job $@ python $(BUILD_DATAUNIT_SCRIPT) --dataunit "$<" --out_data "$@" --out_metadata data/meta/$*$(PARQUET_FILE_EXT)

data/combined: data/combined/all

data/combined/all: $(patsubst %,data/combined/all/%$(PARQUET_FILE_EXT),$(STUDIES)) data/combined/all/all$(PARQUET_FILE_EXT)

data/combined/all/all$(PARQUET_FILE_EXT): $(patsubst %,data/combined/all/%$(PARQUET_FILE_EXT),$(STUDIES))
	hpc/conda-job $@ python $(BUILD_COMBINED_DATA_SCRIPT) --studies $(STUDIES) --out_data "$@"

data/combined/all/%$(PARQUET_FILE_EXT): data/seq/%
	hpc/conda-job $@ python $(BUILD_COMBINED_DATA_SCRIPT) --studies "$*" --out_data "$@"

#-----------------#
# Subsampled data #
#-----------------#

GROUP_SUBSAMPLING_SCRIPT := bin/group_subsampling.py

data/combined/clustered/final/mode_seq/%_single.parquet: data/combined/clustered/final/mode_seq/%.parquet
	hpc/conda-job $@ python $(GROUP_SUBSAMPLING_SCRIPT) --input "$<" --output "$@" --groupby_col "Cluster_ID"

#--------------#
# Grouped Data #
#--------------#

BUILD_GROUPED_DATA_SCRIPT := bin/build_grouped_data.py

data/combined/grouped: data/combined/grouped/v_j_cdr3len

data/combined/grouped/v_j_cdr3len: $(patsubst %,data/combined/grouped/v_j_cdr3len/%,$(STUDIES)) data/combined/grouped/v_j_cdr3len/all

data/combined/grouped/v_j_cdr3len/%: data/combined/all/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ python $(BUILD_GROUPED_DATA_SCRIPT) --data "$<" --out_dir "$@"

#----------------#
# Clustered Data #
#----------------#

BUILD_CLUSTERED_DATA_SCRIPT := bin/build_clustered_data.py
MERGE_CLUSTERED_DATA_SCRIPT := bin/merge_clustered_data.py
BUILD_CLUSTER_CONSENSUS_SEQ_SCRIPT := bin/build_cluster_consensus_seq.py
BUILD_CLUSTER_REPR_SEQ_SCRIPT := bin/build_cluster_representative_seq.py
GROUPS_LIST_FILENAME := groups-list.txt

#data/combined/clustered/Galson_2015a: $(patsubst %,data/combined/clustered/Galson_2015a/%$(PARQUET_FILE_EXT),$(shell cat data/combined/grouped/v_j_cdr3len/Galson_2015a/$(GROUPS_LIST_FILENAME)))

data/combined/clustered/Galson_2016: $(patsubst %,data/combined/clustered/Galson_2016/%$(PARQUET_FILE_EXT),$(shell cat data/combined/grouped/v_j_cdr3len/Galson_2016/$(GROUPS_LIST_FILENAME)))

data/combined/clustered/all: $(patsubst %,data/combined/clustered/all/%$(PARQUET_FILE_EXT),$(shell cat data/combined/grouped/v_j_cdr3len/all/$(GROUPS_LIST_FILENAME)))

data/combined/clustered/final: $(patsubst %,data/combined/clustered/final/%$(PARQUET_FILE_EXT),$(STUDIES)) data/combined/clustered/final/all$(PARQUET_FILE_EXT)

data/combined/clustered/final/centers/%$(PARQUET_FILE_EXT):
	hpc/conda-job $@ python $(MERGE_CLUSTERED_DATA_SCRIPT) --data_dir "data/combined/clustered/centers/$*" --out_data "$@"

data/combined/clustered/final/consensus_seq/%$(PARQUET_FILE_EXT): data/combined/clustered/final/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ python $(BUILD_CLUSTER_CONSENSUS_SEQ_SCRIPT) --data "$<" --out_data "$@"

data/combined/clustered/final/representative_seq/%$(PARQUET_FILE_EXT): data/combined/clustered/final/%$(PARQUET_FILE_EXT) data/combined/clustered/final/consensus_seq/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ python $(BUILD_CLUSTER_REPR_SEQ_SCRIPT) --clustered_data "$<" --consensus_data "$(word 2,$^)" --out_data "$@"

data/combined/clustered/final/%$(PARQUET_FILE_EXT): data/combined/clustered/%
	hpc/conda-job $@ python $(MERGE_CLUSTERED_DATA_SCRIPT) --data_dir "$<" --out_data "$@"

clustered_data: data/combined/clustered/final

data/combined/clustered: $(patsubst %,data/combined/clustered/%,$(STUDIES))

data/combined/clustered/%$(PARQUET_FILE_EXT): #data/combined/grouped/v_j_cdr3len/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ python $(BUILD_CLUSTERED_DATA_SCRIPT) --data "data/combined/grouped/v_j_cdr3len/$*$(PARQUET_FILE_EXT)" --out_data "$@" --out_centroid_data "data/combined/clustered/centers/$*$(PARQUET_FILE_EXT)"

#---------------#
# Cluster modes #
#---------------#

BUILD_CLUSTER_MODES := bin/cluster_mode_sequences.py

data/combined/clustered/final/mode_seq/%$(PARQUET_FILE_EXT): data/combined/clustered/final/%$(PARQUET_FILE_EXT) 
	hpc/conda-job $@ python $(BUILD_CLUSTER_MODES) --clustered_data "$<" --out_data "$@"

#------------#
# Split data #
#------------#

TRAIN_VALID_SPLIT := bin/train_validation_split.py

data/features_data/kmers/neg_subs/%: data/features_data/kmers/%.parquet data/targets_data/clusters/%.parquet data/combined/clustered/final/%.parquet
	hpc/conda-job $@ python $(TRAIN_VALID_SPLIT) --X_data "$<" --y_data "$(word 2,$^)" --clustered_data "$(word 3,$^)" --X_train_data "$@/X_train.parquet" --y_train_data "$@/y_train.parquet" --X_valid_data "data/features_data/kmers/X_valid.parquet" --y_valid_data "data/features_data/kmers/y_valid.parquet"

data/features_data/kmers/both_subs/%: data/features_data/kmers/%.parquet data/targets_data/clusters/%.parquet data/combined/clustered/final/%.parquet
	hpc/conda-job $@ python $(TRAIN_VALID_SPLIT) --X_data "$<" --y_data "$(word 2,$^)" --clustered_data "$(word 3,$^)" --pos_fraction 0.5 --X_train_data "$@/X_train.parquet" --y_train_data "$@/y_train.parquet" --X_valid_data "data/features_data/kmers/X_valid.parquet" --y_valid_data "data/features_data/kmers/y_valid.parquet"

#---------#
# DL Data #
#---------#

TOKENIZE_RAW_DATA := bin/tokenize_raw_data.py

## Tokenize sequences

### Test
#data/RoBERTa/raw/consensus_seq/test: data/RoBERTa/raw/consensus_seq/test/test.in #data/RoBERTa/raw/consensus_seq/test/test.out

data/RoBERTa/raw/mode_seq/test: data/RoBERTa/raw/mode_seq/test/test.in data/RoBERTa/raw/mode_seq/test/test.out

data/RoBERTa/raw/mode_seq_heavy/test: data/RoBERTa/raw/mode_seq_heavy/test/test.in data/RoBERTa/raw/mode_seq_heavy/test/test.out

data/RoBERTa/raw/mode_seq/test/test.in: data/features_data/raw/X_valid.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 5

data/RoBERTa/raw/mode_seq_heavy/test/test.in: data/features_data/raw/X_valid.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

data/RoBERTa/raw_g2016/mode_seq_heavy/test/test.in: data/features_data_g2016/raw/X_valid.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

### Valid
#data/RoBERTa/raw/consensus_seq/valid: data/RoBERTa/raw/consensus_seq/valid/valid.in data/RoBERTa/raw/consensus_seq/valid/valid.out

data/RoBERTa/raw/mode_seq/valid: data/RoBERTa/raw/mode_seq/valid/valid.in data/RoBERTa/raw/mode_seq/valid/valid.out

data/RoBERTa/raw/mode_seq_heavy/valid: data/RoBERTa/raw/mode_seq_heavy/valid/valid.in data/RoBERTa/raw/mode_seq_heavy/valid/valid.out

data/RoBERTa/raw_subject_split/mode_seq_heavy/valid: data/RoBERTa/raw_subject_split/mode_seq_heavy/valid/valid.in data/RoBERTa/raw_subject_split/mode_seq_heavy/valid/valid.out

data/RoBERTa/raw_subject_split/mode_seq/valid: data/RoBERTa/raw_subject_split/mode_seq/valid/valid.in data/RoBERTa/raw_subject_split/mode_seq/valid/valid.out

data/RoBERTa/raw_g2016/mode_seq_heavy/valid: data/RoBERTa/raw_g2016/mode_seq_heavy/valid/valid.in data/RoBERTa/raw_g2016/mode_seq_heavy/valid/valid.out

data/RoBERTa/raw/mode_seq/valid/valid.in: data/features_data/raw/X_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 5

data/RoBERTa/raw/mode_seq_heavy/valid/valid.in: data/features_data/raw/X_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

data/RoBERTa/raw_subject_split/mode_seq_heavy/valid/valid.in: data/features_data/raw_subject_split/X_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

data/RoBERTa/raw_subject_split/mode_seq/valid/valid.in: data/features_data/raw_subject_split/X_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 5

data/RoBERTa/raw_g2016/mode_seq_heavy/valid/valid.in: data/features_data_g2016/raw/X_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

### Train
data/RoBERTa/raw/mode_seq/train/%/train.in: data/features_data/raw/%/X_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 5

data/RoBERTa/raw/mode_seq_heavy/train/%/train.in: data/features_data/raw/%/X_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

data/RoBERTa/raw_subject_split/mode_seq_heavy/train/%/train.in: data/features_data/raw_subject_split/%/X_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

data/RoBERTa/raw_subject_split/mode_seq/train/%/train.in: data/features_data/raw_subject_split/%/X_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 5

data/RoBERTa/raw_g2016/mode_seq_heavy/train/%/train.in: data/features_data_g2016/raw/%/X_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python $(TOKENIZE_RAW_DATA) --input_data "$<" --out_data $@ --input_col_index 3

### Labels
data/RoBERTa/raw/mode_seq/train/%/train.out: data/features_data/raw/%/y_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw/mode_seq_heavy/train/%/train.out: data/features_data/raw/%/y_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw_subject_split/mode_seq_heavy/train/%/train.out: data/features_data/raw_subject_split/%/y_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw_subject_split/mode_seq/train/%/train.out: data/features_data/raw_subject_split/%/y_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw_g2016/mode_seq_heavy/train/%/train.out: data/features_data_g2016/raw/%/y_train.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw/mode_seq/valid/valid.out: data/features_data/raw/y_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw/mode_seq_heavy/valid/valid.out: data/features_data/raw/y_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw_subject_split/mode_seq_heavy/valid/valid.out: data/features_data/raw_subject_split/y_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw_subject_split/mode_seq/valid/valid.out: data/features_data/raw_subject_split/y_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw_g2016/mode_seq_heavy/valid/valid.out: data/features_data_g2016/raw/y_valid_balanced.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw/mode_seq/test/test.out: data/features_data/raw/y_valid.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw/mode_seq_heavy/test/test.out: data/features_data/raw/y_valid.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

data/RoBERTa/raw_g2016/mode_seq_heavy/test/test.out: data/features_data_g2016/raw/y_valid.parquet
	mkdir -p $(dir $@)
	hpc/conda-job $@ python -c "'import pandas as pd; pd.read_parquet(\"$<\").to_csv(\"$@\", header=None, index=None);'"

### Binarize data
#data/RoBERTa/processed/consensus_seq/train: data/RoBERTa/processed/consensus_seq/train/neg_subs data/RoBERTa/processed/consensus_seq/train/both_subs data/RoBERTa/processed/consensus_seq/extra_both_subs

data/RoBERTa/processed/mode_seq/train: data/RoBERTa/processed/mode_seq/train/neg_subs data/RoBERTa/processed/mode_seq/train/both_subs data/RoBERTa/processed/mode_seq/train/extra_both_subs

data/RoBERTa/processed/mode_seq_heavy/train: data/RoBERTa/processed/mode_seq_heavy/train/neg_subs data/RoBERTa/processed/mode_seq_heavy/train/both_subs data/RoBERTa/processed/mode_seq_heavy/train/extra_both_subs

# Whole dataset CDR3
data/RoBERTa/processed/mode_seq/train/all: data/RoBERTa/raw/mode_seq/train/all data/RoBERTa/raw/mode_seq/test data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/test.in \
        --srcdict $(word 3,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/test.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets CDR3 - full training
data/RoBERTa/processed/mode_seq/train/%_full: data/RoBERTa/raw/mode_seq/train/%_full data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --srcdict $(word 2,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets CDR3
data/RoBERTa/processed/mode_seq/train/%: data/RoBERTa/raw/mode_seq/train/% data/RoBERTa/raw/mode_seq/valid data/RoBERTa/raw/mode_seq/test data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/valid.in \
        --testpref $(word 3,$^)/test.in \
        --srcdict $(word 4,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/valid.out \
        --testpref $(word 3,$^)/test.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Whole dataset heavy
data/RoBERTa/processed/mode_seq_heavy/train/all: data/RoBERTa/raw/mode_seq_heavy/train/all data/RoBERTa/raw/mode_seq_heavy/test data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/test.in \
        --srcdict $(word 3,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/test.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets heavy
data/RoBERTa/processed/mode_seq_heavy/train/%: data/RoBERTa/raw/mode_seq_heavy/train/% data/RoBERTa/raw/mode_seq_heavy/valid data/RoBERTa/raw/mode_seq_heavy/test data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/valid.in \
        --testpref $(word 3,$^)/test.in \
        --srcdict $(word 4,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/valid.out \
        --testpref $(word 3,$^)/test.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets heavy - full training
data/RoBERTa/processed/mode_seq_heavy/train/%_full: data/RoBERTa/raw/mode_seq_heavy/train/%_full data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --srcdict $(word 2,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets heavy - subject split
data/RoBERTa/processed/mode_seq_heavy/train/subject_split_%: data/RoBERTa/raw_subject_split/mode_seq_heavy/train/% data/RoBERTa/raw_subject_split/mode_seq_heavy/valid data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/valid.in \
        --srcdict $(word 3,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/valid.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets heavy - subject split - FULL training
data/RoBERTa/processed/mode_seq_heavy/train/subject_split_%_full: data/RoBERTa/raw_subject_split/mode_seq_heavy/train/%_full data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --srcdict $(word 2,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets CDR3 - subject split
data/RoBERTa/processed/mode_seq/train/subject_split_%: data/RoBERTa/raw_subject_split/mode_seq/train/% data/RoBERTa/raw_subject_split/mode_seq/valid data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/valid.in \
        --srcdict $(word 3,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/valid.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# Subsets CDR3 - subject split - FULL training
data/RoBERTa/processed/mode_seq/train/subject_split_%_full: data/RoBERTa/raw_subject_split/mode_seq/train/%_full data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --srcdict $(word 2,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

# G2016 - Subsets heavy
data/RoBERTa/processed_g2016/mode_seq_heavy/train/%: data/RoBERTa/raw_g2016/mode_seq_heavy/train/% data/RoBERTa/raw_g2016/mode_seq_heavy/valid data/RoBERTa/raw_g2016/mode_seq_heavy/test data/RoBERTa/raw/vocab.txt
	hpc/conda-job $@/input0 fairseq-preprocess \
        --trainpref $</train.in \
        --validpref $(word 2,$^)/valid.in \
        --testpref $(word 3,$^)/test.in \
        --srcdict $(word 4,$^) \
        --only-source \
        --workers 28 \
        --destdir $@/input0
	hpc/conda-job $@/label fairseq-preprocess \
        --trainpref $</train.out \
        --validpref $(word 2,$^)/valid.out \
        --testpref $(word 3,$^)/test.out \
        --only-source \
        --workers 28 \
        --destdir $@/label

#----------#
# Features #
#----------#

CONSTRUCT_FP_SCRIPT := bin/construct_fingerprints.py
K := 3

data/features_data/kmers/cdr3_%$(PARQUET_FILE_EXT): data/combined/clustered/final/mode_seq/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ papermill notebooks/data_preprocessing/k-mers.ipynb notebooks/data_preprocessing/k-mers-$*.ipynb -y '"{'SEQUENCES_DATAFRAME_PATH': $<, 'KMERS_DATAFRAME_OUTPUT_PATH': $@, 'K': $(K), 'SEQ_COL_IDX': 5}"'

data/features_data/kmers/heavy_%$(PARQUET_FILE_EXT): data/combined/clustered/final/mode_seq/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ papermill notebooks/data_preprocessing/k-mers.ipynb notebooks/data_preprocessing/k-mers-$*.ipynb -y '"{'SEQUENCES_DATAFRAME_PATH': $<, 'KMERS_DATAFRAME_OUTPUT_PATH': $@, 'K': $(K), 'SEQ_COL_IDX': 3}"'

data/features_data/1mers/%$(PARQUET_FILE_EXT): data/combined/clustered/final/consensus_seq/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ papermill notebooks/data_preprocessing/k-mers.ipynb notebooks/data_preprocessing/1-mers-$*.ipynb -y '"{'SEQUENCES_DATAFRAME_PATH': $<, 'KMERS_DATAFRAME_OUTPUT_PATH': $@, 'K': 1}"'

data/features_data_g2016/fingerprints_cdr3/%.parquet: data/combined/clustered/final/mode_seq/Galson_2016_single.parquet
	hpc/conda-job "$@" python $(CONSTRUCT_FP_SCRIPT) --input "$<" --output "$@" --fp_type "$*" --seq_col cdr3

data/features_data_g2016/kmers_cdr3/%.parquet: data/combined/clustered/final/mode_seq/%.parquet
	hpc/conda-job $@ papermill notebooks/data_preprocessing/k-mers.ipynb notebooks/data_preprocessing/k-mers-$*.ipynb -y '"{'SEQUENCES_DATAFRAME_PATH': $<, 'KMERS_DATAFRAME_OUTPUT_PATH': $@, 'K': $(K), 'SEQ_COL_IDX': 5}"'

#---------#
# Classes #
#---------#

data/targets_data/clusters/%$(PARQUET_FILE_EXT): data/combined/clustered/final/%$(PARQUET_FILE_EXT)
	hpc/conda-job $@ papermill notebooks/data_preprocessing/ConstructClasses.ipynb notebooks/data_preprocessing/ConstructClasses-$*.ipynb -y '"{'INPUT_PATH': $<, 'OUTPUT_PATH': $@, 'HEPB_SPECIF_THRESHOLD': 0.5}"'

#-------------#
# Environment #
#-------------#

## Create local Conda environment in "condaenv" folder
condaenv: environment.yml
	hpc/condaenv $<
	hpc/conda-job $@ pip install -e .
	hpc/conda-job $@ jupyter labextension install @jupyter-widgets/jupyterlab-manager
	hpc/conda-job $@ jupyter lab build --name='OAS-hepB'

## Run JupyterLab locally or on HPC using qsub
lab: condaenv
	hpc/lab
        
#------------------#
# Utility commands #
#------------------#

# Auto-generated help command (make help)
# Adapted from: https://raw.githubusercontent.com/nestauk/patent_analysis/3beebda/Makefile
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

#-----------#
# Notebooks #
#-----------#

notebooks/%.html: notebooks/%.ipynb
	hpc/conda-job $@ jupyter nbconvert $<

#------------------------#
# Parametrized notebooks #
#------------------------#

# Literals
SPACE :=
SPACE +=
COMMA := ,

STUDIES_PYTHON_LIST := [$(subst $(SPACE),$(COMMA) ,$(strip $(STUDIES)))]

notebooks/all/AntibodiesOverview.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/AntibodiesOverview.ipynb $@ -y '"{'STUDIES': $(STUDIES_PYTHON_LIST)}"'

notebooks/%/AntibodiesOverview.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/AntibodiesOverview.ipynb $@ -y '"{'STUDIES': ['$*']}"'

notebooks/%/ClusteringOverview.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/ClusteringOverview.ipynb $@ -p STUDY $*

notebooks/%/kmersOverview.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/kmersOverview.ipynb $@ -y '"{'KMER_DATA_PATH': 'data/features_data/kmers/$*.parquet', 'TARGETS_DATA_PATH': 'data/targets_data/clusters/$*.parquet'}"'

notebooks/%/1mersOverview.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/kmersOverview.ipynb $@ -y '"{'KMER_DATA_PATH': 'data/features_data/1mers/$*.parquet', 'TARGETS_DATA_PATH': 'data/targets_data/clusters/$*.parquet'}"'

# t-SNE
notebooks/representation_visualizations/CircularFingerprints.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/RepresentationVisualization.ipynb $@ -y '"{'X_TRAIN_PATH': 'data/features_data/fingerprints_subj_split/neg_subs/Circular_\(Morgan\)/X_train.parquet', 'X_VALID_PATH': 'data/features_data/fingerprints_subj_split/valid/Circular_\(Morgan\)/X_valid_balanced.parquet', 'Y_TRAIN_PATH': 'data/features_data/fingerprints_subj_split/neg_subs/Circular_\(Morgan\)/y_train.parquet', 'Y_VALID_PATH': 'data/features_data/fingerprints_subj_split/valid/Circular_\(Morgan\)/y_valid_balanced.parquet', 'REPRESENTATION_TYPE': 'Circular fingerprints'}"'

notebooks/representation_visualizations/kmers.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/RepresentationVisualization.ipynb $@ -y '"{'X_TRAIN_PATH': 'data/features_data/kmers/neg_subs/X_train.parquet', 'X_VALID_PATH': 'data/features_data/kmers/X_valid_balanced.parquet', 'Y_TRAIN_PATH': 'data/features_data/kmers/neg_subs/y_train.parquet', 'Y_VALID_PATH': 'data/features_data/kmers/y_valid_balanced.parquet', 'REPRESENTATION_TYPE': 'k-mers'}"'

notebooks/representation_visualizations/HeavyRoBERTa.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/RepresentationVisualization.ipynb $@ -y '"{'X_TRAIN_PATH': 'data/RoBERTa/generated_features/heavy_train.parquet', 'X_VALID_PATH': 'data/RoBERTa/generated_features/heavy_valid.parquet', 'Y_TRAIN_PATH': 'data/features_data/raw_subject_split/neg_subs/y_train.parquet', 'Y_VALID_PATH': 'data/features_data/raw_subject_split/y_valid_balanced.parquet', 'REPRESENTATION_TYPE': 'RoBERTa Heavy'}"'

notebooks/representation_visualizations/CDR3RoBERTa.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/RepresentationVisualization.ipynb $@ -y '"{'X_TRAIN_PATH': 'data/RoBERTa/generated_features/cdr3_train.parquet', 'X_VALID_PATH': 'data/RoBERTa/generated_features/cdr3_valid.parquet', 'Y_TRAIN_PATH': 'data/features_data/raw_subject_split/neg_subs/y_train.parquet', 'Y_VALID_PATH': 'data/features_data/raw_subject_split/y_valid_balanced.parquet', 'REPRESENTATION_TYPE': 'RoBERTa CDR3'}"'

notebooks/representation_visualizations/TestHeavyRoBERTa.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/TestRepresentationVisualization.ipynb $@ -y '"{'X_TEST_PATH': 'data/RoBERTa/generated_features/heavy_test.parquet', 'REPRESENTATION_TYPE': 'RoBERTa Heavy'}"'

notebooks/representation_visualizations/TestCDR3RoBERTa.ipynb:
	hpc/conda-job $@ papermill notebooks/templates/TestRepresentationVisualization.ipynb $@ -y '"{'X_TEST_PATH': 'data/RoBERTa/generated_features/cdr3_test.parquet', 'REPRESENTATION_TYPE': 'RoBERTa CDR3'}"'

#---------------------------#
# DL training - pre-trained #
#---------------------------#

# Weighted CDR3
models/RoBERTa/mode_seq/%/05_weighted_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/% models/RoBERTa/pre-trained/cdr3/checkpoint_best_smooth.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=2 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion weighted_sentence_prediction \
        --class-weights 1 2 \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# CDR3
models/RoBERTa/mode_seq/%/02_reg_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/% models/RoBERTa/pre-trained/cdr3/checkpoint_best_smooth.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-6 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.5 --attention-dropout 0.2 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

models/RoBERTa/mode_seq/%/02_morereg_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/% models/RoBERTa/pre-trained/cdr3/checkpoint_best_smooth.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=1 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.5 --attention-dropout 0.5 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# CDR3
models/RoBERTa/mode_seq/%/02_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/% models/RoBERTa/pre-trained/cdr3/checkpoint_best_smooth.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=2 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Weighted Heavy
models/RoBERTa/mode_seq_heavy/%/04_weighted_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/% models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=2 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion weighted_sentence_prediction \
        --class-weights 1 3 \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy
models/RoBERTa/mode_seq_heavy/%/02_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/% models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy - more regularizaion
models/RoBERTa/mode_seq_heavy/%/04_small_pretrained_reg_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/% models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.2 --attention-dropout 0.2 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy - more regularizaion - ---------------- TMP
models/RoBERTa/mode_seq_heavy/%/04_morereg_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/% models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=1 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.5 --attention-dropout 0.5 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'


# Heavy - Galson_2016
models/RoBERTa/mode_seq_heavy_g2016/%/02_small_pretrained_2000epochs: data/RoBERTa/processed_g2016/mode_seq_heavy/train/% models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

#-------------------------------------------#
# DL training - pre-trained, frozen encoder #
#-------------------------------------------#

# CDR3
models/RoBERTa/mode_seq/%/03_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/% models/RoBERTa/pre-trained/cdr3/checkpoint_best_smooth.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=1 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'


# CDR3 - continue2
models/RoBERTa/mode_seq/subject_split_neg_subs/03_continue2_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/subject_split_neg_subs models/RoBERTa/mode_seq/subject_split_neg_subs/03_frozen_small_pretrained_2000epochs/checkpoints/checkpoint_last.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=2
	hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 6000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy
models/RoBERTa/mode_seq_heavy/%/03_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/% models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=1 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy - continue
models/RoBERTa/mode_seq_heavy/subject_split_neg_subs/03_continue_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs models/RoBERTa/mode_seq_heavy/subject_split_neg_subs/03_frozen_small_pretrained_2000epochs/checkpoints/checkpoint_last.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 4000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy - continue2
models/RoBERTa/mode_seq_heavy/subject_split_neg_subs/03_continue2_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs models/RoBERTa/mode_seq_heavy/subject_split_neg_subs/03_continue_frozen_small_pretrained_2000epochs/checkpoints/checkpoint_last.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=4 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 7000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Weighted Heavy
models/RoBERTa/mode_seq_heavy/%/03_weighted_reg_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/% models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion weighted_sentence_prediction \
        --class-weights 1 2 \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.3 --attention-dropout 0.3 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 6000 \
        --save-interval 10 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

#----------------------#
# Train-test alignment #
#----------------------#

BEST_ALIGNMENTS_SCRIPT := bin/get_best_alignments.py

# POS - POS ---------------------------------
data/train_test_alignments/alignments_cdr3.parquet: data/train_test_alignments/Galson_2015a_pos.parquet data/train_test_alignments/Galson_2016_pos.parquet
	hpc/conda-job $@ python $(BEST_ALIGNMENTS_SCRIPT) --source $(word 2,$^) --target $< --seq_col cdr3 --out_data $@

data/train_test_alignments/alignments_cdr3_part0.parquet: data/train_test_alignments/Galson_2015a_pos_unique_cdr3.parquet data/train_test_alignments/Galson_2016_pos_unique_cdr3_part0.parquet
	hpc/conda-job $@ python $(BEST_ALIGNMENTS_SCRIPT) --source $(word 2,$^) --target $< --seq_col cdr3 --out_data $@

data/train_test_alignments/alignments_cdr3_part1.parquet: data/train_test_alignments/Galson_2015a_pos_unique_cdr3.parquet data/train_test_alignments/Galson_2016_pos_unique_cdr3_part1.parquet
	hpc/conda-job $@ python $(BEST_ALIGNMENTS_SCRIPT) --source $(word 2,$^) --target $< --seq_col cdr3 --out_data $@

# POS - NEG train ----------------------------
data/train_test_alignments/train_alignments_cdr3_part0.parquet: data/train_test_alignments/Galson_2015a_pos_unique_cdr3.parquet data/train_test_alignments/Galson_2015a_neg_unique_cdr3_train_part0.parquet
	hpc/conda-job $@ python $(BEST_ALIGNMENTS_SCRIPT) --source $(word 2,$^) --target $< --seq_col cdr3 --out_data $@

data/train_test_alignments/train_alignments_cdr3_part1.parquet: data/train_test_alignments/Galson_2015a_pos_unique_cdr3.parquet data/train_test_alignments/Galson_2015a_neg_unique_cdr3_train_part1.parquet
	hpc/conda-job $@ python $(BEST_ALIGNMENTS_SCRIPT) --source $(word 2,$^) --target $< --seq_col cdr3 --out_data $@

# POS - NEG ---------------------------------
data/train_test_alignments/pos_neg_alignments_cdr3_part0.parquet: data/train_test_alignments/Galson_2015a_neg_train.parquet data/train_test_alignments/Galson_2016_pos_unique_cdr3_part0.parquet
	hpc/conda-job $@ python $(BEST_ALIGNMENTS_SCRIPT) --source $(word 2,$^) --target $< --seq_col cdr3 --out_data $@

data/train_test_alignments/pos_neg_alignments_cdr3_part1.parquet: data/train_test_alignments/Galson_2015a_neg_train.parquet data/train_test_alignments/Galson_2016_pos_unique_cdr3_part1.parquet
	hpc/conda-job $@ python $(BEST_ALIGNMENTS_SCRIPT) --source $(word 2,$^) --target $< --seq_col cdr3 --out_data $@

#------------------#
# Full DL training # 
#------------------#

#---------------------------#
# DL training - pre-trained #
#---------------------------#

# CDR3
models/RoBERTa/mode_seq/%_full/02_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/%_full models/RoBERTa/pre-trained/cdr3/checkpoint_best_smooth.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=2 hpc/conda-job $@ 'fairseq-train \
        $< \
        --valid-subset train --disable-validation \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy
models/RoBERTa/mode_seq_heavy/%_full/02_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/%_full models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --valid-subset train --disable-validation \
        --user-dir bin/fairseq_plugins \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

#-------------------------------------------#
# DL training - pre-trained, frozen encoder #
#-------------------------------------------#

# CDR3
models/RoBERTa/mode_seq/%_full/03_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq/train/%_full models/RoBERTa/pre-trained/cdr3/checkpoint_best_smooth.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=2
	hpc/conda-job $@ 'fairseq-train \
        $< \
        --valid-subset train --disable-validation \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 32 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

# Heavy
models/RoBERTa/mode_seq_heavy/%_full/03_frozen_small_pretrained_2000epochs: data/RoBERTa/processed/mode_seq_heavy/train/%_full models/RoBERTa/pre-trained/heavy/checkpoint_heavy_seq.pt
	mkdir -p $@
	#rm -rf $@/checkpoints $@/tensorboard
	JOB_GPUS=3 hpc/conda-job $@ 'fairseq-train \
        $< \
        --valid-subset train --disable-validation \
        --user-dir bin/fairseq_plugins \
        --freeze-encoder \
        --init-token 0 --separator-token 2 \
        --restore-file $(word 2,$^) --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir $@/checkpoints \
        --tensorboard-logdir $@/tensorboard \
        --fp16 --ddp-backend=no_c10d \
        --arch roberta_small \
        --criterion sentence_prediction \
        --task sentence_prediction \
        --num-classes 2 \
        --optimizer adam \
        --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 \
        --max-positions 144 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 256 \
        --max-epoch 2000 \
        --log-format simple \
        --log-interval 1000 \
        --validate-interval 1 \
            2>&1 | tee $@/log'

#--------------#
# DL utilities #
#--------------#

BEST_CHECKPOINT_SCRIPT := bin/roberta_full_best_checkpoint.py

models/RoBERTa/mode_seq/subject_split_neg_subs_full/%/checkpoints/checkpoint_best.pt: models/RoBERTa/mode_seq/subject_split_neg_subs/%/checkpoints/checkpoint_best.pt
	hpc/conda-job $@ python $(BEST_CHECKPOINT_SCRIPT) --out_checkpoint_dir "$(dir $@)" --out_checkpoint_filename "$(notdir $@)" --valid_best_checkpoint "$<"

models/RoBERTa/mode_seq_heavy/subject_split_neg_subs_full/%/checkpoints/checkpoint_best.pt: models/RoBERTa/mode_seq_heavy/subject_split_neg_subs/%/checkpoints/checkpoint_best.pt
	hpc/conda-job $@ python $(BEST_CHECKPOINT_SCRIPT) --out_checkpoint_dir "$(dir $@)" --out_checkpoint_filename "$(notdir $@)" --valid_best_checkpoint "$<"

#-----------------------------#
# RoBERTa features generation #
#-----------------------------#

ROBERTA_INTERACT := bin/roberta_interact.py

data/RoBERTa/generated_features: data/RoBERTa/generated_features/cdr3_train.parquet data/RoBERTa/generated_features/cdr3_valid.parquet data/RoBERTa/generated_features/heavy_train.parquet data/RoBERTa/generated_features/heavy_valid.parquet

data/RoBERTa/generated_features/cdr3_train.parquet: data/features_data/raw_subject_split/neg_subs/X_train.parquet models/RoBERTa/mode_seq/subject_split_neg_subs_full/02_small_pretrained_2000epochs/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 100 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq/train/subject_split_neg_subs_full/ \
		--seq_col cdr3 \
		--max_len 30 \
		--output $@ \
		--extract_features \
		--cpu

data/RoBERTa/generated_features/cdr3_valid.parquet: data/features_data/raw_subject_split/X_valid_balanced.parquet models/RoBERTa/mode_seq/subject_split_neg_subs_full/02_small_pretrained_2000epochs/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 100 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq/train/subject_split_neg_subs_full/ \
		--seq_col cdr3 \
		--max_len 30 \
		--output $@ \
		--extract_features \
		--cpu

data/RoBERTa/generated_features/cdr3_test.parquet: data/combined/clustered/final/mode_seq/Galson_2016_single.parquet models/RoBERTa/mode_seq/subject_split_neg_subs_full/02_small_pretrained_2000epochs/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 500 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq/train/subject_split_neg_subs_full/ \
		--seq_col cdr3 \
		--max_len 30 \
		--output $@ \
		--extract_features \
		--cpu

data/RoBERTa/generated_features/heavy_train.parquet: data/features_data/raw_subject_split/neg_subs/X_train.parquet models/RoBERTa/mode_seq_heavy/subject_split_neg_subs_full/02_small_pretrained_2000epochs/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 700 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs_full/ \
		--seq_col seq \
		--output $@ \
		--extract_features \
		--cpu

data/RoBERTa/generated_features/heavy_valid.parquet: data/features_data/raw_subject_split/X_valid_balanced.parquet models/RoBERTa/mode_seq_heavy/subject_split_neg_subs_full/02_small_pretrained_2000epochs/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 700 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs_full/ \
		--seq_col seq \
		--output $@ \
		--extract_features \
		--cpu

data/RoBERTa/generated_features/heavy_test.parquet: data/combined/clustered/final/mode_seq/Galson_2016_single.parquet models/RoBERTa/mode_seq_heavy/subject_split_neg_subs_full/02_small_pretrained_2000epochs/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 1000 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs_full/ \
		--seq_col seq \
		--output $@ \
		--extract_features \
		--cpu

#------------------#
# Test predictions #
#------------------#

ROBERTA_INTERACT := bin/roberta_interact.py

data/RoBERTa/predictions/test: data/RoBERTa/predictions/test_heavy data/RoBERTa/predictions/test_cdr3

data/RoBERTa/predictions/test_heavy: data/RoBERTa/predictions/test_heavy/02_small_pretrained_2000epochs.npy data/RoBERTa/predictions/test_heavy/03_frozen_small_pretrained_2000epochs.npy

data/RoBERTa/predictions/test_cdr3: data/RoBERTa/predictions/test_cdr3/02_small_pretrained_2000epochs.npy data/RoBERTa/predictions/test_cdr3/03_frozen_small_pretrained_2000epochs.npy

data/RoBERTa/predictions/test_heavy/%.npy: data/combined/clustered/final/mode_seq/Galson_2016_single.parquet models/RoBERTa/mode_seq_heavy/subject_split_neg_subs_full/%/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 1000 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs_full/ \
		--seq_col seq \
		--cpu \
		--output $@

data/RoBERTa/predictions/test_cdr3/%.npy: data/combined/clustered/final/mode_seq/Galson_2016_single.parquet models/RoBERTa/mode_seq/subject_split_neg_subs_full/%/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 100 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq/train/subject_split_neg_subs_full/ \
		--seq_col cdr3 \
		--max_len 30 \
		--cpu \
		--output $@

#------------------------#
# Validation predictions #
#------------------------#

data/RoBERTa/predictions: data/RoBERTa/predictions/valid_heavy data/RoBERTa/predictions/valid_balanced_heavy data/RoBERTa/predictions/valid_cdr3 data/RoBERTa/predictions/valid_balanced_cdr3 data/RoBERTa/predictions/test_heavy data/RoBERTa/predictions/test_balanced_heavy data/RoBERTa/predictions/test_cdr3 data/RoBERTa/predictions/test_balanced_cdr3

# Heavy
data/RoBERTa/predictions/valid_heavy: data/RoBERTa/predictions/valid_heavy/02_small_pretrained_2000epochs.npy data/RoBERTa/predictions/valid_heavy/03_frozen_small_pretrained_2000epochs.npy

data/RoBERTa/predictions/valid_balanced_heavy: data/RoBERTa/predictions/valid_balanced_heavy/02_small_pretrained_2000epochs.npy data/RoBERTa/predictions/valid_balanced_heavy/03_frozen_small_pretrained_2000epochs.npy

data/RoBERTa/predictions/valid_heavy/%.npy: data/features_data/raw_subject_split/X_valid.parquet models/RoBERTa/mode_seq_heavy/subject_split_neg_subs/%/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 1000 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs/ \
		--seq_col seq \
		--cpu \
		--output $@

data/RoBERTa/predictions/valid_balanced_heavy/%.npy: data/features_data/raw_subject_split/X_valid_balanced.parquet models/RoBERTa/mode_seq_heavy/subject_split_neg_subs/%/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 300 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq_heavy/train/subject_split_neg_subs/ \
		--seq_col seq \
		--cpu \
		--output $@

# CDR3
data/RoBERTa/predictions/valid_cdr3: data/RoBERTa/predictions/valid_cdr3/02_small_pretrained_2000epochs.npy data/RoBERTa/predictions/valid_cdr3/03_frozen_small_pretrained_2000epochs.npy

data/RoBERTa/predictions/valid_balanced_cdr3: data/RoBERTa/predictions/valid_balanced_cdr3/02_small_pretrained_2000epochs.npy data/RoBERTa/predictions/valid_balanced_cdr3/03_frozen_small_pretrained_2000epochs.npy

data/RoBERTa/predictions/valid_cdr3/%.npy: data/features_data/raw_subject_split/X_valid.parquet models/RoBERTa/mode_seq/subject_split_neg_subs/%/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 100 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq/train/subject_split_neg_subs/ \
		--seq_col cdr3 \
		--max_len 30 \
		--cpu \
		--output $@

data/RoBERTa/predictions/valid_balanced_cdr3/%.npy: data/features_data/raw_subject_split/X_valid_balanced.parquet models/RoBERTa/mode_seq/subject_split_neg_subs/%/checkpoints/
	hpc/conda-job $@ python $(ROBERTA_INTERACT) \
		--input $< \
		--batch_cnt 100 \
		--checkpoint_dir $(word 2,$^) \
		--checkpoint_file checkpoint_best.pt \
		--data ../../../../../../data/RoBERTa/processed/mode_seq/train/subject_split_neg_subs/ \
		--seq_col cdr3 \
		--max_len 30 \
		--cpu \
		--output $@