import torch
from transformers import LongformerModel, LongformerTokenizerFast
from ai4code.models import Model


loss = torch.nn.BCEWithLogitsLoss()


def test_long_former():
    text = "Death is the irreversible cessation of all biological functions that sustain an organism. " \
           "Brain death is sometimes used as a legal definition of death." \
           "The remains of a former organism normally begin to decompose shortly after death. " \
           "Death is an inevitable, universal process that eventually occurs in all organisms"

    tokenizer = LongformerTokenizerFast.from_pretrained("/home/featurize/longformer-base-4096")
    model = LongformerModel.from_pretrained("/home/featurize/longformer-base-4096")
    model.cuda()
    inputs = tokenizer.encode(text)
    with torch.no_grad():
        input_ids = torch.randint(0, high=1000, size=(32, 1024)).cuda()
        model(input_ids)
    print("here")



def test_model_struct():
    model = BaseModel("/home/featurize/distilbert-base-uncased/distilbert-base-uncased")
    for p in model.parameters():
        print(p.requires_grad)


def test_model():
    model = BaseModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00003)
    for i in range(100000):
        input_ids = torch.randint(high=4000, size=(32, 512)).cuda()
        mask = torch.ones_like(input_ids).cuda()
        output = model(input_ids, mask)

        labels = torch.rand(4, 1).cuda()
        l = loss(output, labels)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
