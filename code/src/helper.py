### output solution ###
def save_output(YourName, psp, suffix):
    """save the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        psp::PSP
            an PSP object
        suffix::str
            suffix of the output file,
            'initial' for random initialization
            and 'solution' for the final solution
    """
    generate_output(YourName, psp, suffix)


### generate output file for the solution ###
def generate_output(YourName, psp, suffix):
    """Generate output file (.txt) for the psp solution, containing the instance name, the objective value, and the route
    Args:
        YourName::str
            your name, eg. John_Doe
        psp::PSP
            an PSP object
        suffix::str
            suffix of the output file,
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    """
    workers = sorted(psp.workers, key=lambda x: x.id)
    str_builder = [
        f"Objective: {psp.objective()}, Unassigned: {[t.id for t in psp.unassigned]}"
    ]
    str_builder += [str(w) for w in workers]
    str_builder = [e for e in str_builder if len(e) > 0]
    with open("{}_{}_{}.txt".format(YourName, psp.name, suffix), "w") as f:
        f.write("\n".join(str_builder))
