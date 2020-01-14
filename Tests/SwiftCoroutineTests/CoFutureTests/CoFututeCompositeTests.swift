//
//  CoFututeCompositeTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 13.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFututeCompositeTests: XCTestCase {
    
    func testComposite() {
        func test<T>(futures: [CoFuture<T>],
                     @CoFututeComposite<T> builder: @escaping () -> [CoFuture<T>]) {
            XCTAssertEqual(futures, builder())
        }
        let promises = (0..<3).map { _ in CoPromise<Int>() }
        test(futures: promises) {
            promises[0]
            promises[1]
            promises[2]
        }
    }
    
}
