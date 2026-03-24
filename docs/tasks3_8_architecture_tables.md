# Tasks 3-8: Components, Connectors, and Data Elements

## Scope
This document completes the following continuous assessment tasks:

- Task 3: Identify all components for overall architecture for Design Choice 1
- Task 4: Identify all components for overall architecture for Design Choice 2
- Task 5: Identify all connectors for overall architecture for Design Choice 1
- Task 6: Identify all connectors for overall architecture for Design Choice 2
- Task 7: Identify data element(s) for overall architecture for Design Choice 1
- Task 8: Identify data element(s) for overall architecture for Design Choice 2

The tables below are aligned with the current project and with the assignment rule that `Type 1` is not classified.

## Design Choice 1 Overview
Design Choice 1 uses chained multi-outputs. The system predicts three dependent outputs derived from the same email instance:

- `L1 = Type 2`
- `L2 = Type 2 + Type 3`
- `L3 = Type 2 + Type 3 + Type 4`

The same model family is evaluated on each chained target using one consistent feature representation and one consistent data object.

## Task 3: Components for Design Choice 1
| ID | Component | Existing or Proposed | Responsibility |
| --- | --- | --- | --- |
| DC1-C1 | `main.py` main controller | Existing | Starts the system and triggers the pipeline |
| DC1-C2 | `pipeline.py` chained pipeline controller | Existing, to be extended | Coordinates preprocessing, chain creation, feature generation, model execution, and reporting |
| DC1-C3 | `config.py` configuration component | Existing | Stores shared variables such as column names, random seed, and modeling options |
| DC1-C4 | Data loading component in `preprocessing.py` | Existing | Loads `AppGallery.csv` and `Purchasing.csv`, renames columns, merges datasets |
| DC1-C5 | Text cleaning component in `preprocessing.py` | Existing | Removes duplicates, cleans noise, normalizes text |
| DC1-C6 | Chained target builder | Proposed | Creates `L1`, `L2`, and `L3` from `y2`, `y3`, and `y4` |
| DC1-C7 | `embeddings.py` feature generator | Existing | Converts text into TF-IDF numeric vectors |
| DC1-C8 | `data_loader.py` chained data object | Existing, to be extended | Encapsulates `X_train`, `X_test`, and chained target splits in one consistent object |
| DC1-C9 | `model/base.py` abstract model interface | Existing | Enforces shared methods such as `train()`, `predict()`, and `print_results()` |
| DC1-C10 | `model/random_forest.py` | Existing | Implements Random Forest behind the shared interface |
| DC1-C11 | `model/hist_gb.py` | Existing | Implements Histogram Gradient Boosting behind the shared interface |
| DC1-C12 | `model/sgd.py` | Existing | Implements SGD classifier behind the shared interface |
| DC1-C13 | `model/adaboost.py` | Existing | Implements AdaBoost behind the shared interface |
| DC1-C14 | `model/voting.py` | Existing | Implements Voting classifier behind the shared interface |
| DC1-C15 | `model/random_trees_ens.py` | Existing | Implements Extra Trees behind the shared interface |
| DC1-C16 | Result reporting component | Existing, to be extended | Prints or stores evaluation results for `L1`, `L2`, and `L3` |

## Task 5: Connectors for Design Choice 1
| ID | Source | Destination | Connector Type | Purpose |
| --- | --- | --- | --- | --- |
| DC1-K1 | `main.py` | `pipeline.py` | Method call | Starts the chained multi-output workflow through `run()` |
| DC1-K2 | `pipeline.py` | `config.py` | Import | Reads shared configuration values |
| DC1-K3 | `pipeline.py` | Data loading component | Function call | Loads and merges raw CSV data |
| DC1-K4 | `pipeline.py` | Text cleaning component | Function call | Applies de-duplication and noise removal |
| DC1-K5 | `pipeline.py` | Chained target builder | Function call | Builds `L1`, `L2`, and `L3` labels |
| DC1-K6 | `pipeline.py` | `embeddings.py` | Function call | Generates one numeric feature matrix from cleaned email text |
| DC1-K7 | `pipeline.py` | Chained data object | Object construction | Packages features and labels into one consistent data carrier |
| DC1-K8 | `pipeline.py` | Model classes | Object creation | Instantiates each model implementation |
| DC1-K9 | Model classes | `model/base.py` | Inheritance | Ensures each model exposes the same interface |
| DC1-K10 | `pipeline.py` | Model instance | Polymorphic method call | Calls `train()` through a uniform interface |
| DC1-K11 | `pipeline.py` | Model instance | Polymorphic method call | Calls `predict()` through a uniform interface |
| DC1-K12 | `pipeline.py` | Model instance | Polymorphic method call | Calls `print_results()` through a uniform interface |
| DC1-K13 | Chained data object | Model instance | Object/data passing | Supplies `X_train`, `X_test`, and target splits |
| DC1-K14 | Model instance | Result reporting component | Data return or print | Exposes metrics for each chained label level |

## Task 7: Data Elements for Design Choice 1
| ID | Data Element | Structure | Produced By | Used By | Purpose |
| --- | --- | --- | --- | --- | --- |
| DC1-D1 | Raw CSV files | Two CSV files | External data source | Data loading component | Original email ticket records |
| DC1-D2 | Unified raw dataset | Pandas DataFrame | Data loading component | Text cleaning component | Combined dataset with common schema |
| DC1-D3 | Cleaned dataset | Pandas DataFrame | Text cleaning component | Chained target builder, feature generator | Deduplicated and normalized records |
| DC1-D4 | Text corpus | Series or array of strings | Cleaned dataset | Feature generator | Concatenated ticket summary and interaction content |
| DC1-D5 | `L1` target | Label vector | Chained target builder | Chained data object, models, evaluator | Represents `Type 2` |
| DC1-D6 | `L2` target | Label vector | Chained target builder | Chained data object, models, evaluator | Represents `Type 2 + Type 3` |
| DC1-D7 | `L3` target | Label vector | Chained target builder | Chained data object, models, evaluator | Represents `Type 2 + Type 3 + Type 4` |
| DC1-D8 | Feature matrix `X` | Numeric matrix | Feature generator | Chained data object, models | Shared input representation for all models |
| DC1-D9 | Train/test splits | Train and test arrays | Chained data object | Models | Encapsulates `X_train`, `X_test`, and the target splits for all chain levels |
| DC1-D10 | Rare-class filter result | Label subset metadata | Chained data object | Models, evaluator | Removes labels with too few instances |
| DC1-D11 | Predictions for `L1`, `L2`, `L3` | Label vectors | Model instances | Result reporting component | Stores model outputs for evaluation |
| DC1-D12 | Evaluation results | Metrics table or printed output | Result reporting component | User or report | Compares model performance across chained outputs |

## Design Choice 2 Overview
Design Choice 2 uses hierarchical modeling. The system does not combine labels into one chained label. Instead, it chains model instances.

The flow is:

1. a root model predicts `Type 2`
2. one child model per `Type 2` class predicts `Type 3`
3. one grandchild model per `Type 2` plus `Type 3` path predicts `Type 4`

This design allows model effectiveness to be assessed per parent-class branch.

## Task 4: Components for Design Choice 2
| ID | Component | Existing or Proposed | Responsibility |
| --- | --- | --- | --- |
| DC2-C1 | `main.py` main controller | Existing | Starts the hierarchical workflow |
| DC2-C2 | `pipeline.py` hierarchical controller | Existing, to be extended | Coordinates root, child, and grandchild model execution |
| DC2-C3 | `config.py` configuration component | Existing | Stores shared configuration values |
| DC2-C4 | Data loading component in `preprocessing.py` | Existing | Loads and merges source data |
| DC2-C5 | Text cleaning component in `preprocessing.py` | Existing | Cleans and normalizes email text |
| DC2-C6 | `embeddings.py` feature generator | Existing | Produces one numeric representation of email text |
| DC2-C7 | Root data object for `Type 2` | Existing, to be extended | Holds the first-level training and testing data |
| DC2-C8 | Root model instance | Existing model classes reused | Predicts `Type 2` |
| DC2-C9 | Level-2 filter manager | Proposed | Creates one filtered subset for each `Type 2` class |
| DC2-C10 | Child data objects for `Type 3` | Proposed | Hold one subset per `Type 2` branch |
| DC2-C11 | Child model instances | Existing model classes reused | Predict `Type 3` inside each `Type 2` branch |
| DC2-C12 | Level-3 filter manager | Proposed | Creates one filtered subset for each `Type 2` and `Type 3` path |
| DC2-C13 | Grandchild data objects for `Type 4` | Proposed | Hold one subset per hierarchical path |
| DC2-C14 | Grandchild model instances | Existing model classes reused | Predict `Type 4` inside each `Type 2` and `Type 3` path |
| DC2-C15 | `model/base.py` abstract model interface | Existing | Enforces a consistent model API across root, child, and grandchild models |
| DC2-C16 | Model factory or model registry | Proposed | Creates and stores repeated model instances for each branch |
| DC2-C17 | Hierarchical result reporting component | Proposed | Aggregates path-aware predictions and metrics |

## Task 6: Connectors for Design Choice 2
| ID | Source | Destination | Connector Type | Purpose |
| --- | --- | --- | --- | --- |
| DC2-K1 | `main.py` | `pipeline.py` | Method call | Starts the hierarchical modeling workflow |
| DC2-K2 | `pipeline.py` | `config.py` | Import | Reads shared settings |
| DC2-K3 | `pipeline.py` | Data loading component | Function call | Loads merged CSV data |
| DC2-K4 | `pipeline.py` | Text cleaning component | Function call | Applies preprocessing to raw text |
| DC2-K5 | `pipeline.py` | `embeddings.py` | Function call | Produces the shared feature matrix |
| DC2-K6 | `pipeline.py` | Root data object | Object construction | Packages first-level data for `Type 2` |
| DC2-K7 | `pipeline.py` or model factory | Root model instance | Object creation | Creates the first-level classifier |
| DC2-K8 | Root model instance | `model/base.py` | Inheritance | Keeps the root classifier behind the same interface |
| DC2-K9 | `pipeline.py` | Root model instance | Polymorphic method call | Trains and tests the `Type 2` model |
| DC2-K10 | Root model output | Level-2 filter manager | Data-driven routing | Determines which `Type 3` subset each record enters |
| DC2-K11 | Level-2 filter manager | Child data objects | Object creation | Builds one filtered dataset per `Type 2` class |
| DC2-K12 | Model factory or pipeline | Child model instances | Object creation | Creates one `Type 3` model per `Type 2` branch |
| DC2-K13 | Child model instances | `model/base.py` | Inheritance | Keeps branch-specific models behind the same interface |
| DC2-K14 | `pipeline.py` | Child model instances | Polymorphic method call | Trains and tests each `Type 3` branch model |
| DC2-K15 | Child model output | Level-3 filter manager | Data-driven routing | Determines which `Type 4` subset each record enters |
| DC2-K16 | Level-3 filter manager | Grandchild data objects | Object creation | Builds one filtered dataset per hierarchical path |
| DC2-K17 | Model factory or pipeline | Grandchild model instances | Object creation | Creates one `Type 4` model per path |
| DC2-K18 | Grandchild model instances | `model/base.py` | Inheritance | Keeps path-specific models behind the same interface |
| DC2-K19 | `pipeline.py` | Grandchild model instances | Polymorphic method call | Trains and tests each `Type 4` path model |
| DC2-K20 | Root, child, and grandchild models | Hierarchical result reporting component | Data aggregation | Combines outputs into full hierarchical predictions and metrics |

## Task 8: Data Elements for Design Choice 2
| ID | Data Element | Structure | Produced By | Used By | Purpose |
| --- | --- | --- | --- | --- | --- |
| DC2-D1 | Raw CSV files | Two CSV files | External data source | Data loading component | Original ticket data |
| DC2-D2 | Unified raw dataset | Pandas DataFrame | Data loading component | Text cleaning component | Common source dataset |
| DC2-D3 | Cleaned dataset | Pandas DataFrame | Text cleaning component | Feature generator, filter managers | Preprocessed email records |
| DC2-D4 | Text corpus | Series or array of strings | Cleaned dataset | Feature generator | Text used to create numeric features |
| DC2-D5 | Feature matrix `X` | Numeric matrix | Feature generator | Root and branch data objects | Shared representation across all hierarchy levels |
| DC2-D6 | Root target `y2` | Label vector | Cleaned dataset | Root data object, root model | First-level dependent variable |
| DC2-D7 | Root train/test split | Train and test arrays | Root data object | Root model | First-level train/test data |
| DC2-D8 | Filter Set A | Collection of filtered subsets | Level-2 filter manager | Child data objects, child models | One subset per `Type 2` class for predicting `Type 3` |
| DC2-D9 | Child targets `y3 | y2 = c` | Label vectors | Filter Set A | Child data objects, child models | Second-level targets inside each `Type 2` branch |
| DC2-D10 | Child train/test splits | Train and test arrays | Child data objects | Child models | Per-branch train/test data for `Type 3` |
| DC2-D11 | Filter Set B | Collection of filtered subsets | Level-3 filter manager | Grandchild data objects, grandchild models | One subset per `Type 2` and `Type 3` path for predicting `Type 4` |
| DC2-D12 | Grandchild targets `y4 | y2 = c, y3 = d` | Label vectors | Filter Set B | Grandchild data objects, grandchild models | Third-level targets inside each hierarchical path |
| DC2-D13 | Grandchild train/test splits | Train and test arrays | Grandchild data objects | Grandchild models | Per-path train/test data for `Type 4` |
| DC2-D14 | Model tree or model registry | Mapping structure | Model factory or pipeline | Hierarchical controller | Stores model instance per branch or path |
| DC2-D15 | Routed predictions | Label vectors or tuples | Root, child, and grandchild models | Hierarchical result reporting component | Holds branch-aware outputs from each hierarchy level |
| DC2-D16 | Final hierarchical prediction | Tuple or combined record | Hierarchical result reporting component | User or report | Full prediction across `Type 2`, `Type 3`, and `Type 4` |
| DC2-D17 | Hierarchical evaluation results | Metrics table or printed output | Hierarchical result reporting component | User or report | Measures branch-level and end-to-end performance |

## Submission-Ready Summary
Tasks 3-8 are satisfied by identifying the architecture building blocks for both design choices. For Design Choice 1, the main components are the controller, preprocessing modules, chained target builder, feature generator, shared data object, abstract model interface, concrete model classes, and reporting module. The connectors are the calls, imports, inheritance links, and object-passing relationships that allow one controller to preprocess once, create chained labels once, and evaluate all models through a common interface. The main data elements are the raw and cleaned datasets, TF-IDF feature matrix, chained labels, train and test splits, predictions, and evaluation outputs.

For Design Choice 2, the architecture adds a hierarchy manager made of root, child, and grandchild data objects and model instances, along with filter managers that create branch-specific subsets. The connectors show how the output of one model level routes records into the next level. The main data elements are the shared feature matrix, root target, branch-specific filter sets, branch-level train and test splits, routed predictions, the model tree, and final hierarchical evaluation results.
