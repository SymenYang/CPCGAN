def experience(self):
    total = 0
    times = 100
    for i in range(times):
        total += self.model.run_FPD()
        if i % 10 == 0:
            print(i)
    print("Averaged FPD :",total / times)