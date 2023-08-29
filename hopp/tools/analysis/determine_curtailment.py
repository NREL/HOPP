def determine_curtailment_function(gen_kw, curtailment_limit_kw, verbosity=0):
    curtailment_result = []
    curtailed_gen_list = []
    amount_curtailed_list = []
    for i, gen in enumerate(gen_kw):
        if gen >= curtailment_limit_kw:
            curtailed_gen = curtailment_limit_kw
            amount_curtailed = gen - curtailed_gen
            curtailed_gen_list.append(curtailed_gen)
            amount_curtailed_list.append(amount_curtailed)
            if verbosity == 2:
                print("Curtailed Generation is:", curtailed_gen)
                print("Amount Curtailed was:", amount_curtailed)

        else:
            curtailed_gen = gen
            amount_curtailed = 0
            curtailed_gen_list.append(curtailed_gen)
            amount_curtailed_list.append(amount_curtailed)
            if verbosity == 2:
                print("Curtailed Generation is:", curtailed_gen)

    if verbosity == 2:
        print(curtailed_gen_list)
        print(amount_curtailed_list)
    raw_gen_total = sum(gen_kw)
    curtailed_gen_total = sum(curtailed_gen_list)
    amount_curtailed_total = sum(amount_curtailed_list)
    percentage_curtailment = 100 * amount_curtailed_total / raw_gen_total

    if verbosity >= 1:
        print("Curtailed generation signal is: ", curtailed_gen_list)
        print("Amount curtailed is: ", amount_curtailed_list)
        print("Total amount curtailed: ", amount_curtailed_total)
        print("Percentage curtailment was: {:f}%".format(percentage_curtailment))

    return curtailed_gen_list, curtailed_gen_total, amount_curtailed_list, amount_curtailed_total, raw_gen_total, percentage_curtailment
