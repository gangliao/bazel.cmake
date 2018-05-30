#! /usr/bin/env python

# See README.txt for information and build instructions.

import addressbook_pb2
import sys

try:
  raw_input          # Python 2
except NameError:
  raw_input = input  # Python 3


# This function fills in a Person message based on user input.
def PromptForAddress(person):
  print("Enter person ID number: 10")
  person.id = 10
  print("Enter name: Gang Liao")
  person.name = "Gang Liao"

  print("Enter email address (blank for none): gangliao@umd.edu")
  person.email = "gangliao@umd.edu"

  print("Enter a phone number (or leave blank to finish): 10")

  phone_number = person.phones.add()
  phone_number.number = "10101010"

  print("Is this a mobile, home, or work phone? mobile")
  type = "mobile"
  if type == "mobile":
    phone_number.type = addressbook_pb2.Person.MOBILE
  elif type == "home":
    phone_number.type = addressbook_pb2.Person.HOME
  elif type == "work":
    phone_number.type = addressbook_pb2.Person.WORK
  else:
    print("Unknown phone type; leaving as default value.")


# Main procedure:  Reads the entire address book from a file,
#   adds one person based on user input, then writes it back out to the same
#   file.
if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "ADDRESS_BOOK_FILE")
  sys.exit(-1)

address_book = addressbook_pb2.AddressBook()

# Read the existing address book.
try:
  with open(sys.argv[1], "rb") as f:
    address_book.ParseFromString(f.read())
except IOError:
  print(sys.argv[1] + ": File not found.  Creating a new file.")

# Add an address.
PromptForAddress(address_book.people.add())

# Write the new address book back to disk.
with open(sys.argv[1], "wb") as f:
  f.write(address_book.SerializeToString())