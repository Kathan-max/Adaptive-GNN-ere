import json
import re
import ast

data = """698.2s 3058 spouse :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 2}
698.2s 3059 followed by :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3060 use :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3061 based on :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3062 has part :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3063 part of :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3064 point in time :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3065 follows :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3066 subclass of :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3067 publication date :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.3, 'heads': 2}
698.2s 3068 instance of :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3069 country :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.3, 'heads': 2}
698.2s 3070 field of work :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3071 capital :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.2, 'heads': 4}
698.2s 3072 opposite of :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3073 different from :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3074 shares border with :  {'model_type': 'GIN', 'hidden_channels': 64, 'dropout': 0.2, 'heads': None}
698.2s 3075 present in work :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3076 characters :  {'model_type': 'GIN', 'hidden_channels': 64, 'dropout': 0.2, 'heads': None}
698.2s 3077 discoverer or inventor :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3078 father :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3079 child :  {'model_type': 'GCN', 'hidden_channels': 64, 'dropout': 0.3, 'heads': None}
698.2s 3080 facet of :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3081 student :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3082 publisher :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 2}
698.2s 3083 number of articles :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.2, 'heads': 2}
698.2s 3084 length :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3085 point in :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3086 published in :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3087 said to be the same as :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3088 has effect :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3089 has cause :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.2, 'heads': 4}
698.2s 3090 number of participants :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3091 author :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 2}
698.2s 3092 main subject :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3093 named after :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3094 influenced by :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3095 instance of <triplet>parenleftBig :  {'model_type': 'GIN', 'hidden_channels': 64, 'dropout': 0.3, 'heads': None}
698.2s 3096 located in the administrative territorial entity :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.2, 'heads': 4}
698.2s 3097 educated at :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3098 studies :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3099 practiced by :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3100 area :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3101 subsidiary :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3102 parent organization :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.3, 'heads': 2}
698.2s 3103 religious order :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3104 located in the administrative :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3105 has parts of the class :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 2}
698.2s 3106 diplomatic relation :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.2, 'heads': 4}
698.2s 3107 religion :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3108 work period (start) :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.3, 'heads': 4}
698.2s 3109 used by :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3110 developer :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3111 continent :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3112 inception :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3113 location of formation :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3114 owned by :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 2}
698.2s 3115 industry :  {'model_type': 'GraphSAGE', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3116 twinned administrative body :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3117 product or material produced :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3118 employer :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3119 number of episodes :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 2}
698.2s 3120 elevation above sea level :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3121 founded by :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.2, 'heads': 4}
698.2s 3122 dissolved, abolished or demolished date :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3123 member of :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3124 place of birth :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3125 notable work :  {'model_type': 'GCN', 'hidden_channels': 64, 'dropout': 0.3, 'heads': None}
698.2s 3126 creator :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.2, 'heads': 2}
698.2s 3127 is a list of :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3128 headquarters location :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3129 country of citizenship :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3130 applies to jurisdiction :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3131 place of publication :  {'model_type': 'GIN', 'hidden_channels': 64, 'dropout': 0.3, 'heads': None}
698.2s 3132 sibling :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3133 event distance :  {'model_type': 'GIN', 'hidden_channels': 64, 'dropout': 0.2, 'heads': None}
698.2s 3134 work location :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.3, 'heads': None}
698.2s 3135 replaces :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.3, 'heads': 2}
698.2s 3136 basin country :  {'model_type': 'GIN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3137 population :  {'model_type': 'GraphSAGE', 'hidden_channels': 64, 'dropout': 0.3, 'heads': None}
698.2s 3138 programming language :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3139 member of sports team :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.3, 'heads': 4}
698.2s 3140 end time :  {'model_type': 'GAT', 'hidden_channels': 64, 'dropout': 0.3, 'heads': 4}
698.2s 3141 country of origin :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 2}
698.2s 3142 derivative work :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3143 number of cores :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 2}
698.2s 3144 designed by :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.2, 'heads': 4}
698.2s 3145 terminus :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
698.2s 3146 family :  {'model_type': 'GAT', 'hidden_channels': 128, 'dropout': 0.3, 'heads': 4}
698.2s 3147 location :  {'model_type': 'GIN', 'hidden_channels': 64, 'dropout': 0.3, 'heads': None}
698.2s 3148 ranking :  {'model_type': 'GCN', 'hidden_channels': 128, 'dropout': 0.2, 'heads': None}
"""

pattern = r"\d+\.\ds \d+ (.+?) :  (\{.*\})"

# Extract matches
matches = re.findall(pattern, data)
# print(matches[0])

# Convert matches into dictionary
result_dict = {key.strip(): ast.literal_eval(value.strip()) for key, value in matches}

# # Print result
# print(json.dumps(result_dict, indent=2))
with open("../Results/Best_arch_report/Law.json", "w") as f:
    json.dump(result_dict, f, indent=4)