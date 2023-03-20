#include"model.h"
#include"optimizer.h"
#include"DNN.h"

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<float.h>

#define Data_Length 579
typedef struct _Data
{
    double input[2], label;
}Data;

int epoch = 5000, iter;
double lr = 1e-2;
extern loss_func criterion;
extern Model *model, *output_layer;
extern Optim_param update_param;

double predict(Mat *);

int main(int *argc, char **argv)
{
    FILE *fp;
    Data data[Data_Length];
    int index;
    double total_loss, pre;
    srand(time(NULL));

    fp = fopen("579_norm.txt", "r");
    for(index = 0; index < Data_Length; index++)
    {
        fscanf(fp, "%lf", &data[index].input[0]);
        fscanf(fp, "%lf", &data[index].input[1]);
        fscanf(fp, "%lf", &data[index].label);
    }
    fclose(fp);

    DNN();
    Loss("Cross_Entropy");
    opt("SGD");
    update_param.learning_rate = lr;
    update_param.momentum = 0.0;
    update_param.weight_decay = 0.0;

    fp = fopen("loss.csv", "w");
    for(iter = 0; iter < epoch; iter++)
    {
        total_loss = 0.0;
        for(index = 0; index < Data_Length; index++)
        {
            forward_pass(data[index].input);
            total_loss += criterion(output_layer->layer->output, data[index].label - 1.0);
            backward_pass();

            update();
        }
        total_loss /= Data_Length;
        fprintf(fp, "%d,%lf\n", iter, total_loss);
    }
    fclose(fp);

    total_loss = 0.0;
    for(index = 0; index < Data_Length; index++)
    {
        forward_pass(data[index].input);
        pre = predict(output_layer->layer->output);
        printf("%d\t%lf\n", index, pre);
        if(data[index].label == pre)
            total_loss++;
    }
    printf("acc:%lf%%\n", total_loss / Data_Length * 100.0);

    return 0;
}

double predict(Mat *output)
{
    int index, pre;
    double max = -DBL_MAX, value;

    for(index = 0; index < output->row * output->col; index++)
    {
        value = output->element[index];
        if(max < value)
        {
            max = value;
            pre = index + 1;
        }
    }

    return (double)pre;
}