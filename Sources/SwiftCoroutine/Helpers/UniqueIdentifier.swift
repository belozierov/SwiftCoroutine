//
//  UniqueIdentifier.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 01.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@usableFromInline
struct UniqueIdentifier: Hashable {
    
    private static var counter = AtomicInt(wrappedValue: .min)
    let id = counter.increase()
    
}
