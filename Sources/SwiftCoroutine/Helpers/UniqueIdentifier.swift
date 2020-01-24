//
//  UniqueIdentifier.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 23.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

struct UniqueIdentifier: Hashable {
    
    private static var counter = AtomicInt(wrappedValue: .min)
    private let id = counter.increase()
    
}
