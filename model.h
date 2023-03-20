#ifndef MODEL_H
#define MODEL_H

#include"matrix.h"
#include"nn.h"
//#include"optimizer.h"

typedef struct _Layer Layer;
typedef struct _Model Model;
typedef void (* pass)(Layer *, int);
typedef void (* act_func)(Mat *);
typedef double (* loss_func)(Mat *, int);
//typedef void (*optim)(Model *, Optim_param);


struct _Layer
{
    Mat *input, *output;
    Mat *weight, *bias, *weight_gradient, *bias_gradient;
    act_func act, act_gradient;
    pass forward, backward;
};

struct _Model
{
    Layer *layer;
    Model *forward_link, *backward_link;
};


void push_back(Model *);
//void clear_Model(Model *);
void forward_pass(double *);
void backward_pass(void);
void update(void);
void xavier_init(Mat *, int);


void Linear(int, int, int);
void Conv(int, int, int *, int *, int *, int *, int);
void Activation(const char *);
void Loss(const char *);
void Linear_forward(Layer *, int);
void Linear_backward(Layer *, int);
void opt(const char *);

#endif