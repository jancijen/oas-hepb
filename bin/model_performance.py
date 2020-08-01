from bin.prediction import predict_in_batches

def model_performance(model_tuple, data, metric_fns, sample_weights, verbose, batches_cnt=5):
    model_name, model = model_tuple
    X_train, X_valid, y_train, y_valid = data

    try:
        if verbose:
            print(f'Training {model_name}...')

        # Fit the model
        model.fit(X_train, y_train)

        if verbose:
            print(f'Predicting using {model_name}...')

        # Validation performance
        y_pred = predict_in_batches(model, X_valid.values, batches_cnt=batches_cnt)

        validation_weights = sample_weights.loc[y_valid.index] if sample_weights is not None else None
        metric_vals = [(metric_name, metric_fn(y_valid, y_pred, sample_weight=validation_weights)) for metric_name, metric_fn in metric_fns]
        non_weighted_metric_vals = [(f'Non-weighted {metric_name}', metric_fn(y_valid, y_pred)) for metric_name, metric_fn in metric_fns]
        metric_vals.extend(non_weighted_metric_vals)

        if verbose:
            metric_vals_str = ', '.join(['{}: {:.3f}'.format(metric_name, metric_val) for metric_name, metric_val in metric_vals])
            print(f'{model_name} - {metric_vals_str}\n')
    except ValueError as error:
        print(f'{model_name}: {error}\n')
        return None
        
    return model_tuple, metric_vals, y_pred

def model_selection(models, data, metric_fns, sample_weights=None, verbose=True):
    if verbose:
        print(f'Metric values:\n')
    
    model_performances = {}
    trained_models = {}
    for model in models:
        model_tuple, metric_vals, _ = model_performance(model, data, metric_fns, sample_weights, verbose)
        model_perf = (model_tuple, metric_vals[0])
        if model_perf:
            model_performances[model_perf[0][0]] = model_perf[1]
            trained_models[model_perf[0][0]] = model_perf[0][1]

    # Get information about the best performing model on the data
    best_perf = max(model_performances.items(), key=lambda x: x[1][1])
    print('-' * 30)
    print('-' * 30)
    print(f'Best performing model is {best_perf[0]} with metric value ({best_perf[1][0]}) = {"{:.3f}".format(best_perf[1][1])}')

    return best_perf, model_performances, trained_models
