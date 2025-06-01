 """
 Create a Python program that implements a singly linked list using Object-Oriented Programming (OOP) principles. 
Your implementation should include the following: A Node class to represent each node in the list. 
A LinkedList class to manage the nodes, with methods to: Add a node to the end of the list Print the list Delete the nth node (where n is a 1-based index) 
Include exception handling to manage edge cases such as: Deleting a node from an empty list Deleting a node
with an index out of range Test your implementation with at least one sample list.
"""


class Node:
    """A single node in the linked list, holding data and a reference to the next node."""
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    """A linked list class that manages nodes and provides basic operations."""
    def __init__(self):
        self.head = None

    def append(self, data):
        """Adds a node with the given data to the end of the list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node

    def print_list(self):
        """Prints all nodes in the list, separated by arrows."""
        if self.head is None:
            print("The list is empty.")
            return
        curr = self.head
        while curr:
            print(curr.data, end=" -> ")
            curr = curr.next
        print("None")

    def delete_nth_node(self, n):
        """Deletes the node at the 1-based index n, with error handling."""
        try:
            if self.head is None:
                raise Exception("Cannot delete from an empty list.")
            if n <= 0:
                raise IndexError("Index must be a positive integer (1-based).")
            if n == 1:
                print(f"Deleting node at position {n} with value {self.head.data}")
                self.head = self.head.next
                return
            curr = self.head
            prev = None
            count = 1
            while curr and count < n:
                prev = curr
                curr = curr.next
                count += 1
            if curr is None:
                raise IndexError("Index out of range.")
            print(f"Deleting node at position {n} with value {curr.data}")
            prev.next = curr.next
        except Exception as e:
            print("Error:", e)

# ====== Test the implementation ======
if __name__ == "__main__":
    ll = LinkedList()
    ll.append(10)
    ll.append(20)
    ll.append(30)
    ll.append(40)
    print("Initial Linked List:")
    ll.print_list()

    ll.delete_nth_node(2)
    print("\nLinked List after deleting 2nd node:")
    ll.print_list()

    ll.delete_nth_node(10)
    ll.delete_nth_node(0)
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)
    ll.delete_nth_node(1)
