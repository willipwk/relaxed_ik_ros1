import urdf_parser_py.urdf as urdf

def get_chains(joint, robot, chain=None):
    if chain is None:
        chain = []

    
    chain.append(joint)
    chain.append(joint.child)

    chains = []
    for joint in robot.joints:
        if joint.parent == chain[-1]:
            new_chains = get_chains(joint, robot, chain[:])
            chains.extend(new_chains)

    if len(chains) == 0:
        return [chain]

    return chains

def print_chains(chains, print_links=False):
    for i, chain in enumerate(chains):
        print(f"Chain {i + 1}:")
        for element in chain:
            if isinstance(element, str):
                if print_links:
                    print(element)
            else:
                print(element.name, element.type)

def print_chains_joints(chains):
    for i, chain in enumerate(chains):
        print(f"Chain {i + 1}:")
        for element in chain:
            if isinstance(element, str):
                pass
            else:
                print(f'{element.name},')
                
def main():
    robot = urdf.URDF.from_xml_file('/home/ubuntu/rangedik_project/src/relaxed_ik_ros1/relaxed_ik_core/configs/urdfs/simplified_movo.urdf')

    # Assuming the root joint is the first joint in the list
    
    root_joint = robot.joints[0]
    # print(root_joint)

    chains = get_chains(root_joint, robot)
    print_chains(chains, True)
    # print_chains_joints(chains)
    print('\n\n')
    print(',\n'.join([f"'{j.name}'" for j in robot.joints]))
    print(len(robot.joints))

if __name__ == "__main__":
    main()