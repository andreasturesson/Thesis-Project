from matplotlib import pyplot as plt

## hämta data ifrån log

contents = open("Logs/train_gan_2.log", "r").read().split("\n")

gen_losses = []
disc_losses = []
val_accs = []
print(contents[0])

for c in contents:
    print(c.split(","))
    epoch, gen_loss, disc_loss, val_acc = c.split(",")

    gen_losses.append(float(gen_loss))
    disc_losses.append(float(disc_loss))
    val_accs.append(float(val_acc))

plt.figure(1)
plt.plot(val_accs, label='training')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=16)
plt.legend()
plt.figure(2)
plt.plot(gen_losses, label='generator')
plt.plot(disc_losses, label='discriminator')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=16)
plt.legend()
plt.show()