class Item:
    def calculate_total_price(self, x, y):
        return x * y
    
item1 = Item()
item1.name = "Phone"
item1.price = 100
item1.quantity = 5
print(item1.calculate_total_price(item1.price, item1.quantity))

item2 = Item()
item2.name = "Laptop"
item2.price = 1000
item2.quantity = 3
print(item2.calculate_total_price(item2.price, item2.quantity))

##############################################################################################
import os
os.chdir("C:/Users/massw/OneDrive/Bureau/Programmation")
import csv

class Item:
    pay_rate = 0.8 # class attribute # Pay 
    all = []
    def __init__(self, name:str, price: float, quantity=0):
        # Run validations to the received arguments
        assert price >= 0, f'Price {price} is negative !'
        assert quantity >= 0, f'Quantity {quantity} is negative !'
        
        # Assign to self object
        self.name = name
        self.price = price
        self.quantity = quantity

        # Actions to execute
        Item.all.append(self) # Add all item to a list

    def calculate_total_price(self):
        return self.price * self.quantity
    
    def apply_discount(self):
        self.price = self.price * self.pay_rate

    @classmethod
    def instantiate_from_csv(cls):
        with open('OOP_Course.csv', 'r') as f:
            reader = csv.DictReader(f)
            items = list(reader)

        for item in items:
            Item(
                name=item.get('name'),
                price=float(item.get('Price')),
                quantity=float(item.get('Quantity'))
            )

    def __repr__(self):
        return f"Item('{self.name}',{self.price},{self.quantity})"

item1 = Item('Phone', 100, 1)
item2 = Item('Laptop', 1000, 3)

print(item1.calculate_total_price())
print(item2.calculate_total_price())

print(Item.__dict__) # all attributes for class level
print(item1.__dict__) # all the attributes for instances level

# APPLY PAY_RATE DISCOUNT ON ALL INSTANCES
item1.apply_discount()
print(item1.price)

# APPLY SPECIFIC DISCOUNT ON ONE INSTANCE
item2 = Item('Laptop', 1000, 3)
item2.pay_rate = 0.7
item2.apply_discount()
print(item2.price)

# Print all the name instances
for instance in Item.all:
    print(instance.name)

# Print all items on csv like a dictionary
Item.instantiate_from_csv()
print(Item.all)