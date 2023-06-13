import omni.replicator.core as rep

with rep.new_layer():

    #texture randomization
    def texture_rand():
        shapes = rep.get.prims(semantics=[
            ('class', 'can0'), ('class', 'can1'), ('class', 'can2'), ('class', 'can3'), 
            ('class', 'bottle0'), ('class', 'bottle1'), ('class', 'bottle2'), ('class', 'bottle3'), 
            ('class', 'bag0'), ('class', 'bag1'), ('class', 'bag2'), ('class', 'bag3'), 
            ('class', 'box0'),('class', 'box1'),('class', 'box2'),('class', 'box3'),
            ('class', 'cup0'),('class', 'cup1'),('class', 'cup2'),('class', 'cup3'),
            ('class', 'opener0'),('class', 'opener1'),('class', 'opener2'),('class', 'opener3'),
            ('class', 'can30'), ('class', 'can31'), ('class', 'can32'), ('class', 'can33'),
            ('class', 'cap0'),('class', 'cap1'),('class', 'cap2'),('class', 'cap3')])
        # print(type(shapes))
        with shapes:
            rep.randomizer.texture(textures=[
                    '/home/walker2/houdini19.5/models/coke_can/tex/Coca_Cola_by_al_xx.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/Cuivre_3_50.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/metal005.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/stainless_aluminium.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/mpm_vol.09_p35_bottle_cap_diff.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/mpm_vol.09_p35_steel.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/vm_v2_044_bottle1_diffuse.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/vm_v2_044_bottle2_diffuse.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/vm_v2_044_bottle3_diffuse.jpg',
                    '/home/walker2/replicator/texture/pexels-lisa-fotios-3703737.jpg',
                    '/home/walker2/replicator/texture/pexels-polina-kovaleva-6787400.jpg',
                    '/home/walker2/replicator/texture/VCG41N1412923888.jpg',
                    '/home/walker2/replicator/texture/pngtree-garbage-texture-background-mottled-decorative-pattern-image_720642.jpg',
                    '/home/walker2/replicator/texture/red-black-brush-stroke-banner-background-perfect-canva.jpg',
                    '/home/walker2/replicator/texture/creative-background-with-wrinkled-paper-effect.jpg',
                    '/home/walker2/replicator/texture/grunge-cracked-wall.jpg',
                    '/home/walker2/replicator/texture/creative-background-with-rough-painted-texture.jpg',
                    '/home/walker2/replicator/texture/texture-background(1).jpg',
                    '/home/walker2/replicator/texture/texture-background.jpg',
                    '/home/walker2/replicator/texture/close-up-pink-crumpled-paper.jpg',
                    '/home/walker2/replicator/texture/old-crumpled-parchment-paper-texture.jpg',
                    '/home/walker2/replicator/texture/closeup-rusty-grunge-wall.jpg',
                    '/home/walker2/replicator/texture/wrinkled-plastic-wrap-texture-black-wallpaper.jpg',
                    '/home/walker2/houdini19.5/models/coke_can/tex/mpm_vol.09_p35_bottle_red_diff.jpg'
                    ])
        return shapes.node

    rep.randomizer.register(texture_rand)

    #light source randomization
    def light_rand(num):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(35000, 5000),
            position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
            scale=rep.distribution.uniform(50, 100),
            count=num
        )
        return lights.node

    rep.randomizer.register(light_rand)

    #pose randomization
    def pose_rand():
        shapes = rep.get.prims(semantics=[
            ('class', 'can0'), ('class', 'can1'), ('class', 'can2'), ('class', 'can3'), 
            ('class', 'bottle0'), ('class', 'bottle1'), ('class', 'bottle2'), ('class', 'bottle3'), 
            ('class', 'bag0'), ('class', 'bag1'), ('class', 'bag2'), ('class', 'bag3'), 
            ('class', 'box0'),('class', 'box1'),('class', 'box2'),('class', 'box3'),
            ('class', 'cup0'),('class', 'cup1'),('class', 'cup2'),('class', 'cup3'),
            ('class', 'opener0'),('class', 'opener1'),('class', 'opener2'),('class', 'opener3'),
            ('class', 'can30'), ('class', 'can31'), ('class', 'can32'), ('class', 'can33'),
            ('class', 'cap0'),('class', 'cap1'),('class', 'cap2'),('class', 'cap3')])
        with shapes:
            rep.modify.pose(
               	position=rep.distribution.uniform((-20, -60, 0), (20, 60, 0)),
                #position=rep.distribution.sequence([(10, 0.0, 0), (20, 0, 0), (30, 0.0, 0.0)]),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 0)),
                # scale=rep.distribution.normal(1, 0.5)
            )
        return shapes.node

    rep.randomizer.register(pose_rand)


    # Setup randomization
    with rep.trigger.on_frame(num_frames=100, interval=1):
        rep.randomizer.texture_rand()
        rep.randomizer.light_rand(10)
        rep.randomizer.pose_rand()