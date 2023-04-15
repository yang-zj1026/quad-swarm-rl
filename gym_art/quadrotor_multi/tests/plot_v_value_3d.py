import plotly.express as px
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

"""
Check plot_v_value_2d.py for more details. But code you should use below

tmp_score=[]
idx=[]
idy=[]
for i in range(-10, 11):
    ti_score = []
    for j in range(-10, 11):
        normalized_obs_dict['obs'][0][0]=i * 0.2
        normalized_obs_dict['obs'][0][1]=j * 0.2
        
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)

        ti_score.append(result['values'].item())            
        idx.append(i * 0.2)
        idy.append(j * 0.2)

    tmp_score.append(ti_score)

print(tmp_score)
print(idx)
print(idy)
"""

x=np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.4000000000000001, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.2000000000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.6000000000000001, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.2000000000000002, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.4000000000000001, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
y=np.array([-2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0, -2.0, -1.8, -1.6, -1.4000000000000001, -1.2000000000000002, -1.0, -0.8, -0.6000000000000001, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2000000000000002, 1.4000000000000001, 1.6, 1.8, 2.0])
value = np.array([[-0.9693396091461182, -0.8801171779632568, -0.8108367919921875, -0.7603106498718262, -0.7232669591903687, -0.6916850805282593, -0.6585843563079834, -0.6276825070381165, -0.6091716289520264, -0.6068120002746582, -0.6234579682350159, -0.656946063041687, -0.6887145042419434, -0.7095425128936768, -0.7258913516998291, -0.7429983615875244, -0.7620010375976562, -0.7822784185409546, -0.8042833805084229, -0.831002414226532, -0.865764856338501], [-0.8987733721733093, -0.8159864544868469, -0.7526195645332336, -0.7073512673377991, -0.6751775741577148, -0.6484270095825195, -0.619480550289154, -0.5905078649520874, -0.5726046562194824, -0.5707920789718628, -0.5880073308944702, -0.6217854022979736, -0.6526832580566406, -0.6713348627090454, -0.6842904686927795, -0.6968417167663574, -0.7108592987060547, -0.7272502779960632, -0.7478638887405396, -0.7757958173751831, -0.8134271502494812], [-0.8407058715820312, -0.763916015625, -0.7058473229408264, -0.6653012633323669, -0.6375754475593567, -0.6153086423873901, -0.5903680324554443, -0.563156008720398, -0.5453567504882812, -0.5435911417007446, -0.5610702633857727, -0.5950268507003784, -0.6249005198478699, -0.6411089897155762, -0.6502442955970764, -0.6576787233352661, -0.6664698123931885, -0.6797343492507935, -0.7006193399429321, -0.7313027381896973, -0.7725111842155457], [-0.7935147285461426, -0.721051812171936, -0.6666311025619507, -0.6296505928039551, -0.6057778596878052, -0.5878335237503052, -0.5671155452728271, -0.5420323014259338, -0.5243322849273682, -0.5224287509918213, -0.5400473475456238, -0.5742741823196411, -0.6031686067581177, -0.616765022277832, -0.6219086647033691, -0.6246705651283264, -0.629938006401062, -0.6425364017486572, -0.6655610799789429, -0.6998258233070374, -0.7448087930679321], [-0.7574135065078735, -0.6863905191421509, -0.632108211517334, -0.5956019163131714, -0.5734886527061462, -0.558996319770813, -0.5428500175476074, -0.5210750102996826, -0.504530668258667, -0.5030955076217651, -0.5211139917373657, -0.5558871030807495, -0.5839119553565979, -0.5947002172470093, -0.5962105989456177, -0.5962961912155151, -0.6012971997261047, -0.615804135799408, -0.6419249773025513, -0.6798725128173828, -0.728585958480835], [-0.7343019247055054, -0.662692129611969, -0.6046802997589111, -0.5639066696166992, -0.5390605926513672, -0.5246256589889526, -0.5116727948188782, -0.49410784244537354, -0.4806281328201294, -0.4811687469482422, -0.5001263618469238, -0.5355939865112305, -0.5628241300582886, -0.5711323022842407, -0.571041464805603, -0.5721422433853149, -0.5800824761390686, -0.5978748798370361, -0.6272150278091431, -0.6686736941337585, -0.7208231687545776], [-0.7234984636306763, -0.6513961553573608, -0.5884690284729004, -0.5401402711868286, -0.5080562829971313, -0.4884399175643921, -0.4736994504928589, -0.45766258239746094, -0.4475160837173462, -0.4514353275299072, -0.47182536125183105, -0.5079313516616821, -0.5353496670722961, -0.5438855886459351, -0.5465263724327087, -0.552299976348877, -0.5648009777069092, -0.5862035751342773, -0.6186858415603638, -0.6633174419403076, -0.7180031538009644], [-0.7210979461669922, -0.6488922834396362, -0.5823547840118408, -0.526321530342102, -0.485262393951416, -0.4574834108352661, -0.43600356578826904, -0.4149976968765259, -0.4028010368347168, -0.40771353244781494, -0.4290335178375244, -0.4668656587600708, -0.4989529848098755, -0.5135098099708557, -0.5229542851448059, -0.534868061542511, -0.5521357655525208, -0.5772119760513306, -0.6127017140388489, -0.6595945358276367, -0.7151485681533813], [-0.7220938801765442, -0.6498624682426453, -0.5804133415222168, -0.517359733581543, -0.466721773147583, -0.4297541379928589, -0.40007591247558594, -0.3702906370162964, -0.3485654592514038, -0.34623825550079346, -0.36498141288757324, -0.4078497886657715, -0.4521139860153198, -0.4778863191604614, -0.49552786350250244, -0.5134937763214111, -0.5355677008628845, -0.5644968748092651, -0.6025171279907227, -0.6502406001091003, -0.7053029537200928], [-0.7257238626480103, -0.6558520793914795, -0.5840890407562256, -0.5133451223373413, -0.450569748878479, -0.4003862142562866, -0.35885167121887207, -0.3172570466995239, -0.28193461894989014, -0.267065167427063, -0.2798212766647339, -0.3281043767929077, -0.3878237009048462, -0.42790770530700684, -0.4555472135543823, -0.4809530973434448, -0.5090676546096802, -0.5424346923828125, -0.5827034711837769, -0.6310049295425415, -0.6865605115890503], [-0.732353687286377, -0.6680165529251099, -0.5986042618751526, -0.5236407518386841, -0.44919323921203613, -0.38263630867004395, -0.3242682218551636, -0.26628220081329346, -0.21463358402252197, -0.18789517879486084, -0.19480633735656738, -0.24310088157653809, -0.31279098987579346, -0.3678549528121948, -0.4096883535385132, -0.44652414321899414, -0.4828023910522461, -0.5209567546844482, -0.5635086297988892, -0.6134895086288452, -0.6717706918716431], [-0.7382638454437256, -0.6790344715118408, -0.6161797642707825, -0.5458967685699463, -0.4700329303741455, -0.3939850330352783, -0.3205677270889282, -0.24624478816986084, -0.18024146556854248, -0.14473199844360352, -0.14509999752044678, -0.1860595941543579, -0.2530043125152588, -0.3130166530609131, -0.3659360408782959, -0.41475653648376465, -0.45984113216400146, -0.5033895969390869, -0.549915075302124, -0.6042559146881104, -0.6678622961044312], [-0.7439861297607422, -0.6888316869735718, -0.6333849430084229, -0.5724977254867554, -0.5047588348388672, -0.4309213161468506, -0.3503032922744751, -0.26269662380218506, -0.18444883823394775, -0.14114868640899658, -0.13525187969207764, -0.16863596439361572, -0.22796034812927246, -0.2822556495666504, -0.3340102434158325, -0.3855327367782593, -0.4343423843383789, -0.4821721315383911, -0.5345494747161865, -0.5963572263717651, -0.6678907871246338], [-0.7550859451293945, -0.7035195827484131, -0.6542675495147705, -0.601350724697113, -0.5415623188018799, -0.47383618354797363, -0.39541149139404297, -0.30557334423065186, -0.22104156017303467, -0.16924118995666504, -0.155889630317688, -0.18328261375427246, -0.23660600185394287, -0.28313958644866943, -0.3264347314834595, -0.3722459077835083, -0.4196140766143799, -0.4691445827484131, -0.5251615047454834, -0.5916053056716919, -0.6679478883743286], [-0.7750188112258911, -0.7256902456283569, -0.6803935170173645, -0.6326950788497925, -0.5777056217193604, -0.5134091377258301, -0.4394714832305908, -0.35762155055999756, -0.27829277515411377, -0.2241271734237671, -0.20512330532073975, -0.22649526596069336, -0.2721734046936035, -0.3086264133453369, -0.3410545587539673, -0.37889397144317627, -0.42403292655944824, -0.475909948348999, -0.5354655981063843, -0.6037983894348145, -0.6793814897537231], [-0.8032724857330322, -0.7547563314437866, -0.7113863229751587, -0.6668450832366943, -0.6148086786270142, -0.5521749258041382, -0.4823116064071655, -0.41087114810943604, -0.34229791164398193, -0.2933894395828247, -0.275989294052124, -0.29455339908599854, -0.33070623874664307, -0.35488438606262207, -0.3764759302139282, -0.40652549266815186, -0.4481539726257324, -0.5009510517120361, -0.5637602806091309, -0.6339111328125, -0.707084059715271], [-0.8380391597747803, -0.789940595626831, -0.74726402759552, -0.7044547200202942, -0.6547496318817139, -0.594882607460022, -0.5307698249816895, -0.4692307710647583, -0.4103732109069824, -0.3667052984237671, -0.35244596004486084, -0.3719538450241089, -0.4035118818283081, -0.42070603370666504, -0.43509209156036377, -0.4578288793563843, -0.4932734966278076, -0.5430169105529785, -0.6055740118026733, -0.6744552850723267, -0.7428025007247925], [-0.8780637979507446, -0.8312147855758667, -0.7891132831573486, -0.7472847700119019, -0.6999979019165039, -0.6455408930778503, -0.5897877216339111, -0.536503791809082, -0.48303890228271484, -0.4400843381881714, -0.42636191844940186, -0.4478602409362793, -0.47965383529663086, -0.49751222133636475, -0.5114096403121948, -0.5303771495819092, -0.559801459312439, -0.6032946109771729, -0.6596469879150391, -0.721369743347168, -0.7821807861328125], [-0.9244278073310852, -0.8802056312561035, -0.8395999670028687, -0.7987861633300781, -0.7541112899780273, -0.7063811421394348, -0.6591713428497314, -0.6115380525588989, -0.5599044561386108, -0.5149868726730347, -0.5007821321487427, -0.5248538255691528, -0.5576447248458862, -0.5771971940994263, -0.5931128263473511, -0.6131638288497925, -0.6411207914352417, -0.6788331270217896, -0.7255969047546387, -0.7767788171768188, -0.8289562463760376], [-0.9820675849914551, -0.9413673877716064, -0.9030877351760864, -0.8633398413658142, -0.8206771612167358, -0.7781949043273926, -0.7373519539833069, -0.6943734884262085, -0.6454242467880249, -0.600406289100647, -0.5863338708877563, -0.6111443042755127, -0.6415500640869141, -0.6598657369613647, -0.6767855882644653, -0.6989148855209351, -0.7270010709762573, -0.760886549949646, -0.8007103204727173, -0.8438501358032227, -0.8888550996780396], [-1.0549429655075073, -1.0172317028045654, -0.9805482625961304, -0.9413827657699585, -0.9005316495895386, -0.8622474670410156, -0.8268524408340454, -0.7899229526519775, -0.7470723390579224, -0.7054203748703003, -0.6917507648468018, -0.713533878326416, -0.737011194229126, -0.7511458396911621, -0.7672113180160522, -0.7887042760848999, -0.8138699531555176, -0.8436121940612793, -0.8796828985214233, -0.9195690155029297, -0.9614615440368652]]).flatten()


def plot_v_value_map_2d(x, y, value, width=480, height=480, show=False):
    dd = {'x': x, 'y': y, 'value': value}
    df = pd.DataFrame(dd)
    max_v_id = df['value'].idxmax()
    max_x, max_y, max_value = df.iloc[max_v_id]['x'], df.iloc[max_v_id]['y'], df.iloc[max_v_id]['value']

    text = "max value={:.5f}, x={:.2f}, y={:.2f}".format(max_value, max_x, max_y)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    sc = ax.scatter(x, y, c=value, cmap='viridis')
    cbar = fig.colorbar(sc, ax=ax)
    plt.title(text)

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape((width, height, 3))
    # img = fig.to_image(format="png", width=width, height=height)

    if show:
        plt.show()

    return img_array


if __name__ == '__main__':
    plot_v_value_map_3d(x, y, value, show=True)