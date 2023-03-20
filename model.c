#include"model.h"
#include"optimizer.h"

#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>

typedef void (*optim)(Layer *, Optim_param);

Model *model = NULL;
Model *output_layer = NULL;
loss_func criterion;
optim optimizer;
Optim_param update_param;
int train = 1;

void push_back(Model *new_layer)
{
    new_layer->forward_link = NULL;
    if(model)
    {
        output_layer->forward_link = new_layer;
        new_layer->backward_link = output_layer;
        output_layer = output_layer->forward_link;
    }
    else
    {
        output_layer = model = new_layer;
        model->backward_link = NULL;
    }
}

void Linear(int input_size, int output_size, int bias)
{
    Model *new_layer = (Model *)malloc(sizeof(Model));
    new_layer->layer = (Layer *)malloc(sizeof(Layer));

    new_layer->layer->input = new_matrix(1, input_size);
    new_layer->layer->output = new_matrix(1, output_size);
    new_layer->layer->weight = new_matrix(input_size, output_size);
    new_layer->layer->forward = Linear_forward;
    new_layer->layer->backward = Linear_backward;

    xavier_init(new_layer->layer->weight, input_size);

    new_layer->layer->bias = bias ? new_matrix(1, output_size) : NULL;
    new_layer->layer->weight_gradient = train ? new_matrix(input_size, output_size) : NULL;
    new_layer->layer->bias_gradient = bias && train ? new_matrix(1, output_size) : NULL;

    xavier_init(new_layer->layer->bias, input_size);

    push_back(new_layer);
}

void Activation(const char *act_f)
{
    if(strcmp(act_f, "sigmoid") == 0)
    {
        output_layer->layer->act = Sigmoid;
        output_layer->layer->act_gradient = Sigmoid_gradient;
    }
    else if(strcmp(act_f, "relu") == 0)
    {
        output_layer->layer->act = Relu;
        output_layer->layer->act_gradient = Relu_gradient;
    }
    else if(strcmp(act_f, "softmax") == 0)
    {
        output_layer->layer->act = Softmax;
        output_layer->layer->act_gradient = NULL;
    }
}

void Loss(const char *loss)
{
    if(strcmp(loss, "Cross_Entropy") == 0)
        criterion = Cross_Entropy;
}

void Linear_forward(Layer *layer, int dc)
{
    Mat *temp_1 = NULL, *temp_2 = NULL;

    temp_1 = matrix_product(layer->input, layer->weight);
    if(layer->bias)
    {
        temp_2 = matrix_addtion(layer->bias, temp_1);
        copy_matrix(temp_1, temp_2);
        delete_matrix(temp_2);
    }
    copy_matrix(layer->output, temp_1);
    layer->act(layer->output);
    delete_matrix(temp_1);
}

void Linear_backward(Layer *layer, int sw)
{
    Mat *temp_1 = NULL, *temp_2 =NULL;

    temp_1 = transpose(layer->input);
    temp_2 = matrix_product(temp_1, layer->output);
    copy_matrix(layer->weight_gradient, temp_2);
    delete_matrix(temp_1);
    delete_matrix(temp_2);
    if(layer->bias)
    {
        temp_1 = element_product(layer->bias, layer->output);
        copy_matrix(layer->bias_gradient, temp_1);
        delete_matrix(temp_1);
    }
    if(sw)
    {
        temp_1 = transpose(layer->weight);
        temp_2 = matrix_product(layer->output, temp_1);
        if(layer->act_gradient)
            layer->act_gradient(layer->input);
        delete_matrix(temp_1);
        temp_1 = element_product(layer->input, temp_2);
        copy_matrix(layer->input, temp_1);
        delete_matrix(temp_1);
        delete_matrix(temp_2);
    }
}

void forward_pass(double *input)
{
    Model *ptr = model;
    int index;

    for(index = 0; index < ptr->layer->input->col; index++)
        ptr->layer->input->element[index] = input[index];
    
    for(ptr = model; ptr; ptr = ptr->forward_link)
    {
        ptr->layer->forward(ptr->layer, 1);
        if(ptr->forward_link)
            copy_matrix(ptr->forward_link->layer->input, ptr->layer->output);
    }
}

void backward_pass(void)
{
    Model *ptr;
    int sw;

    for(ptr = output_layer; ptr; ptr = ptr->backward_link)
    {
        sw = ptr->backward_link ? 1 : 0;
        ptr->layer->backward(ptr->layer, sw);
        if(sw)
            copy_matrix(ptr->backward_link->layer->output, ptr->layer->input);
    }
}

void opt(const char *opt_f)
{
    if(strcmp(opt_f, "SGD") == 0)
        optimizer = SGD;
}

void update(void)
{
    Model *ptr;

    for(ptr = model; ptr; ptr = ptr->forward_link)
        optimizer(model->layer, update_param);
}

void xavier_init(Mat *mat, int input_size)
{
    int index;
    double k = 1.0 / sqrt((double)input_size);

    for(index = 0; index < mat->row * mat->col; index++)
        mat->element[index] = 2 * k * rand() / RAND_MAX - k;
}