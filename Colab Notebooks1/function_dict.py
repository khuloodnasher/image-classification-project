#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to create a dictionary of model's train, validation and test results


def store_results_to_dict(model, model_description):
    train_loss, train_acc = model.evaluate_generator(train_generator, steps=train_steps_per_epoch)
    val_loss, val_acc = model.evaluate_generator(val_generator, steps=val_steps_per_epoch)
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_steps_per_epoch)
    pred = model.predict_generator(test_generator, test_steps_per_epoch)
    pred_classes = np.argmax(pred, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    precision_score = precision_score(true_classes, pred_classes)
    recall_score = recall_score(true_classes, pred_classes)
    f1_score = f1_score(true_classes, pred_classes)

    curr_dict = { 'Model':model_description
                 ,'Train Accuracy': train_acc
                 ,'Train Loss': train_loss
                 ,'Validation Accuracy':val_acc
                 ,'validation Loss':val_loss
                 ,'Test Accuracy':test_acc
                 ,'Test Loss':test_loss
                 ,'Precision':precision_score
                 ,'Recall':recall_score
                 ,'f1':f1_score
                  }
    return curr_dict






