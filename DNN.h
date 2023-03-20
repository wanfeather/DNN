#include"model.h"

void DNN(void)
{
    Linear(2, 3, 1);
    Activation("relu");
    Linear(3, 3, 1);
    Activation("relu");
    Linear(3, 3, 1);
    Activation("softmax");
}