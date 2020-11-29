'''
    Find percentage of overlap between Tiny ImageNet and ImageNet-A labels.

    Results:
        74 / 200 of Tiny ImageNet labels are in ImageNet-A, 0.37
        Tiny ImageNet # of labels: 200
        ImageNet-A # of labels: 200
'''

with open("datasets/TinyImageNet_ids.txt", 'r') as file:
    tiny_imgnet_ids = file.read().split('\n')

with open("datasets/ImageNetA_ids.txt", 'r') as file:
    lines = file.read().split('\n')
    imgnet_a_ids = dict([tuple(line.split(' ', 1)) for line in lines])

count = 0
overlap = []
print('Overlapping IDs:')
for id in tiny_imgnet_ids:
    if id in imgnet_a_ids:
        count += 1
        print(id, imgnet_a_ids[id])
        overlap.append(id)

print('Tiny ImageNet # of labels:', len(tiny_imgnet_ids))
print('ImageNet-A # of labels:', len(imgnet_a_ids))
print('% of Tiny ImageNet IDs in ImageNet-A', count, '/',
      len(tiny_imgnet_ids), count / len(tiny_imgnet_ids))

''' 
Overlapping IDs:
n04067472 reel 
n04540053 volleyball 
n04099969 rocking chair 
n07749582 lemon 
n01641577 American bullfrog 
n02802426 basketball 
n09246464 cliff 
n03891332 parking meter 
n02106662 German Shepherd Dog 
n02279972 monarch butterfly 
n04146614 school bus 
n04507155 umbrella 
n03854065 organ 
n03804744 nail 
n02486410 baboon 
n01944390 snail 
n04275548 spider web 
n07695742 pretzel 
n01774750 tarantula 
n07753592 banana 
n02233338 cockroach 
n02236044 mantis 
n07583066 guacamole 
n04456115 torch 
n01855672 goose 
n01882714 koala 
n02669723 academic gown 
n02165456 ladybug 
n02099601 Golden Retriever 
n02948072 candle 
n02206856 bee 
n02814860 lighthouse 
n01910747 jellyfish 
n04133789 sandal 
n02268443 dragonfly 
n07734744 mushroom 
n04562935 water tower 
n03014705 chest 
n02190166 fly 
n03670208 limousine 
n04366367 suspension bridge 
n03026506 Christmas stocking 
n02906734 broom 
n01770393 scorpion 
n04118538 rugby ball 
n04179913 sewing machine 
n02123394 Persian cat 
n02793495 barn 
n02730930 apron 
n03388043 fountain 
n02837789 bikini 
n04399382 teddy bear 
n03355925 flagpole 
n03250847 drumstick 
n03255030 dumbbell 
n02883205 bow tie 
n01698640 American alligator 
n01784675 centipede 
n04376876 syringe 
n03444034 go-kart 
n04532670 viaduct 
n07768694 pomegranate 
n02999410 chain 
n03617480 kimono 
n02410509 bison 
n02226429 grasshopper 
n02231487 stick insect 
n02085620 Chihuahua 
n02129165 lion 
n03837869 obelisk 
n02815834 beaker 
n07720875 bell pepper 
n12267677 acorn 
n02504458 African bush elephant 
'''
