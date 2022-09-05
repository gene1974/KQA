import kopl
from kopl.test.test_example import *

# run_test()
from kopl.kopl import KoPLEngine
from kopl.test.test_example import example_kb

engine = KoPLEngine(example_kb)

ans = engine.SelectBetween(
        engine.Find('LeBron James Jr.'),
        engine.Relate(
                engine.Find('LeBron James Jr.'),
                'father',
                'forward'
        ),
        'height',
        'greater'
)

# func = getattr(engine, 'Relate')
# func = getattr(engine, 'Find')
# inputs = ['LeBron James Jr.']
# ans = func(*inputs)
print(engine.FindAll())
print(engine.FilterNum(engine.FindAll(), 'inflation rate', '4200 percentage', ''))
# print(engine.Find('LeBron James Jr.'))
# print(engine.Relate(
#                 engine.Find('LeBron James Jr.'),
#                 'father',
#                 'forward'
#         ))
# print(engine.SelectBetween(
#         engine.Find('LeBron James Jr.'),
#         engine.Relate(
#                 engine.Find('LeBron James Jr.'),
#                 'father',
#                 'forward'
#         ),
#         'height',
#         'greater'
# ))
print(engine.QueryRelation(
    engine.Find("Viggo Mortensen"),
    engine.Find("10th Screen Actors Guild Awards")
))

print(ans)

# python -m Bart_Program.train --input_dir ../base_data/ --output_dir ../base_checkpoint/ --model_name_or_path ../bart-base/ --save_dir ../base_log/